import base64
from mimetypes import guess_type
import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import csv
import datetime
import re
import requests

# Load environment variables from a .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI()

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
SERVER_API_KEY = os.getenv("SERVER_API_KEY", "change_me_please")
CAMERA_ID = os.getenv("CAMERA_ID", "cam-001")

CSV_FILE = 'detected_plates.csv'
NO_DETECTADO = "No detectado"
CSV_HEADERS = ['placa', 'marca', 'modelo', 'color', 'tipo_carroceria', 'fecha', 'imagen']
METADATA_FIELDS = ['placa', 'marca', 'modelo', 'color', 'tipo_carroceria']

def initialize_csv(file_path):
    if not os.path.isfile(file_path):
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADERS)
        return

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = list(csv.reader(csvfile))

    if not reader:
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(CSV_HEADERS)
        return

    header = reader[0]
    if header == CSV_HEADERS:
        return

    # Rewrite file with new headers, preserving old data where possible
    with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADERS)
        for row in reader[1:]:
            placa = row[0] if len(row) > 0 else NO_DETECTADO
            fecha = row[1] if len(row) > 1 else ''
            imagen = row[2] if len(row) > 2 else ''
            writer.writerow([placa or NO_DETECTADO, NO_DETECTADO, NO_DETECTADO, NO_DETECTADO, NO_DETECTADO, fecha, imagen])

initialize_csv(CSV_FILE)

# Function to encode a local image into a Data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    try:
        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"The image file '{image_path}' was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the image file: {e}")

    # Construct the Data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def write_to_csv(metadata, image_filename):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        metadata.get('placa', NO_DETECTADO),
        metadata.get('marca', NO_DETECTADO),
        metadata.get('modelo', NO_DETECTADO),
        metadata.get('color', NO_DETECTADO),
        metadata.get('tipo_carroceria', NO_DETECTADO),
        date_time,
        image_filename
    ]
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)

def upload_to_server(metadata, image_path):
    url = f"{SERVER_URL}/upload"
    headers = {"X-API-Key": SERVER_API_KEY}
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'plate_number': metadata.get('placa', NO_DETECTADO),
                'brand': metadata.get('marca', NO_DETECTADO),
                'model': metadata.get('modelo', NO_DETECTADO),
                'color': metadata.get('color', NO_DETECTADO),
                'body_type': metadata.get('tipo_carroceria', NO_DETECTADO),
                'camera_id': CAMERA_ID
            }
            response = requests.post(url, headers=headers, files=files, data=data)
            
        if response.status_code == 200:
            print(f"Successfully uploaded plate {metadata.get('placa', NO_DETECTADO)} to server.")
        else:
            print(f"Failed to upload to server: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error uploading to server: {e}")

def is_valid_plate(plate):
    # Example pattern: 3 letters followed by 3 digits
    pattern = r'^[A-Z]{3}\d{3}$'
    return re.match(pattern, plate) is not None

def sanitize_value(value):
    if value is None:
        return NO_DETECTADO
    cleaned = str(value).strip()
    return cleaned if cleaned else NO_DETECTADO

def normalize_metadata(plate_data):
    metadata = {}
    for field in METADATA_FIELDS:
        metadata[field] = sanitize_value(plate_data.get(field, NO_DETECTADO))

    placa = metadata.get('placa', NO_DETECTADO)
    if placa != NO_DETECTADO:
        placa = placa.replace(' ', '').upper()
        metadata['placa'] = placa if placa else NO_DETECTADO

    return metadata

def print_license_plate(filename):

    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the image file name
    image_filename = filename

    # Construct the full path to the image
    image_path = os.path.join(script_dir, image_filename)

    try:
        # Convert the local image to a Data URL
        data_url = local_image_to_data_url(image_path)
    except Exception as error:
        print(error)
        return  # Exit if image processing fails

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analiza la imagen y devuelve un JSON en español con la información del vehículo detectado. Usa exactamente esta estructura (sin texto adicional ni ```):\n{\n  \"placa\": \"ABC123\",\n  \"marca\": \"Suzuki\",\n  \"modelo\": \"Swift\",\n  \"color\": \"Blanco\",\n  \"tipo_carroceria\": \"Hatchback\"\n}\nSi no puedes determinar un campo, usa \"No detectado\" como valor. No inventes guiones en la placa ni añadas comentarios."},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ],
                }
            ],
            max_tokens=50,
            temperature=0
        )
        response_content = response.choices[0].message.content.strip()

        #Intentar parsear el JSON de la respuesta
        try:
            plate_data = json.loads(response_content)
            metadata = normalize_metadata(plate_data)
            plate_number = metadata.get('placa', NO_DETECTADO)
            print(f"Placa detectada: {plate_number}")

            if plate_number != NO_DETECTADO and is_valid_plate(plate_number):
                write_to_csv(metadata, image_filename)
                upload_to_server(metadata, image_path)

            return metadata
        except json.JSONDecodeError:
            print("Error al decodificar el JSON de la respuesta.")
            print(f"Respuesta recibida: {response_content}")
            return {field: NO_DETECTADO for field in METADATA_FIELDS}
        
    except Exception as e:
        # Handle other possible errors
        print(f"An unexpected error occurred: {e}")
