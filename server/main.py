import os
import sqlite3
import asyncio
import json
from datetime import datetime, timedelta
from typing import Annotated
from contextlib import asynccontextmanager

import jwt
import bcrypt
from fastapi import FastAPI, File, Form, UploadFile, Request, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

load_dotenv()

# Configuration
API_KEY = os.getenv("SERVER_API_KEY", "change_me_please")
JWT_SECRET = os.getenv("JWT_SECRET", "change_me_jwt_secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "plates.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# Password hashing with bcrypt directly
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

# SSE broadcast queue
sse_clients: list[asyncio.Queue] = []

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

NO_DETECTADO = "No detectado"


# Pydantic models
class UserCreate(BaseModel):
    nombre: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    token: str
    user: dict


# Database
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            brand TEXT DEFAULT "No detectado",
            model TEXT DEFAULT "No detectado",
            color TEXT DEFAULT "No detectado",
            body_type TEXT DEFAULT "No detectado",
            camera_id TEXT DEFAULT "cam-001",
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            nombre TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# JWT Helpers
def create_token(user_id: int, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


# Auth Dependencies
async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid API Key")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, nombre FROM users WHERE id = ?", (payload["sub"],))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return dict(user)


# SSE Broadcast
async def broadcast_detection(detection: dict):
    for queue in sse_clients:
        await queue.put(detection)


# Auth Endpoints
@app.post("/api/auth/register", response_model=TokenResponse)
async def register(user: UserCreate):
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 6 caracteres")
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE email = ?", (user.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    
    password_hash = hash_password(user.password)
    cursor.execute(
        "INSERT INTO users (email, nombre, password_hash) VALUES (?, ?, ?)",
        (user.email, user.nombre, password_hash)
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    
    token = create_token(user_id, user.email)
    return {"token": token, "user": {"id": str(user_id), "email": user.email, "nombre": user.nombre}}


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, nombre, password_hash FROM users WHERE email = ?", (credentials.email,))
    user = cursor.fetchone()
    conn.close()
    
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    token = create_token(user["id"], user["email"])
    return {"token": token, "user": {"id": str(user["id"]), "email": user["email"], "nombre": user["nombre"]}}


@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return user


# Plates Endpoints
@app.get("/api/plates")
async def get_plates(
    user: dict = Depends(get_current_user),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    camera_id: str | None = None,
    placa: str | None = None
):
    conn = get_db()
    cursor = conn.cursor()
    
    query = "SELECT * FROM plates WHERE 1=1"
    params = []
    
    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if placa:
        query += " AND plate_number LIKE ?"
        params.append(f"%{placa}%")
    
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    plates = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT COUNT(*) as total FROM plates")
    total = cursor.fetchone()["total"]
    conn.close()
    
    return {"plates": plates, "total": total}


@app.delete("/api/plates/{plate_id}")
async def delete_plate(plate_id: int, user: dict = Depends(get_current_user)):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT image_filename FROM plates WHERE id = ?", (plate_id,))
    plate = cursor.fetchone()

    if not plate:
        conn.close()
        raise HTTPException(status_code=404, detail="Plate not found")

    cursor.execute("DELETE FROM plates WHERE id = ?", (plate_id,))
    conn.commit()
    conn.close()

    image_path = os.path.join(UPLOAD_DIR, plate["image_filename"])
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
        except OSError:
            pass

    return {"status": "deleted", "id": plate_id}


# SSE Endpoint
@app.get("/api/stream")
async def stream_detections(token: str):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    queue: asyncio.Queue = asyncio.Queue()
    sse_clients.append(queue)
    
    async def event_generator():
        try:
            while True:
                detection = await queue.get()
                yield {"event": "detection", "data": json.dumps(detection)}
        except asyncio.CancelledError:
            sse_clients.remove(queue)
            raise
    
    return EventSourceResponse(event_generator())


# Helper
def normalize_field(value: str | None) -> str:
    if value is None:
        return NO_DETECTADO
    cleaned = value.strip()
    return cleaned if cleaned else NO_DETECTADO


# Upload Endpoint (for YOLO detector)
@app.post("/upload")
async def upload_plate(
    plate_number: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    brand: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    color: Annotated[str | None, Form()] = None,
    body_type: Annotated[str | None, Form(alias="body_type")] = None,
    camera_id: Annotated[str | None, Form()] = "cam-001",
    api_key: str = Depends(get_api_key)
):
    normalized_plate = normalize_field(plate_number).upper()
    normalized_brand = normalize_field(brand)
    normalized_model = normalize_field(model)
    normalized_color = normalize_field(color)
    normalized_body_type = normalize_field(body_type)
    normalized_camera_id = camera_id or "cam-001"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{normalized_plate}_{timestamp}_{image.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        content = await image.read()
        buffer.write(content)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO plates (plate_number, image_filename, brand, model, color, body_type, camera_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (normalized_plate, filename, normalized_brand, normalized_model, normalized_color, normalized_body_type, normalized_camera_id)
    )
    conn.commit()
    plate_id = cursor.lastrowid
    
    cursor.execute("SELECT * FROM plates WHERE id = ?", (plate_id,))
    new_plate = dict(cursor.fetchone())
    conn.close()

    await broadcast_detection(new_plate)

    return {"status": "success", "id": plate_id, "filename": filename, "plate": normalized_plate}


# Legacy HTML endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plates ORDER BY timestamp DESC")
    plates = cursor.fetchall()
    conn.close()
    return templates.TemplateResponse("index.html", {"request": request, "plates": plates})
