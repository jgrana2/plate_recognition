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

CSV_FILE = 'detected_plates.csv'

def initialize_csv(file_path):
    if not os.path.isfile(file_path):
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Plate Number', 'Date and Time', 'Image Filename'])

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

def write_to_csv(plate_number, image_filename):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([plate_number, date_time, image_filename])

def upload_to_server(plate_number, image_path):
    url = f"{SERVER_URL}/upload"
    headers = {"X-API-Key": SERVER_API_KEY}
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'plate_number': plate_number}
            response = requests.post(url, headers=headers, files=files, data=data)
            
        if response.status_code == 200:
            print(f"Successfully uploaded plate {plate_number} to server.")
        else:
            print(f"Failed to upload to server: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error uploading to server: {e}")

def is_valid_plate(plate):
    # Example pattern: 3 letters followed by 3 digits
    pattern = r'^[A-Z]{3}\d{3}$'
    return re.match(pattern, plate) is not None

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
                        {"type": "text", "text": "Por favor, extrae y devuelve el número de la placa del vehículo visible en la imagen proporcionada. El resultado debe estar formateado como un objeto JSON con la siguiente estructura exacta:\n\n{\n  \"placa\": \"ABC123\"\n}\n\nSi no se detecta ninguna placa, retorna:\n\n{\n  \"placa\": \"No Detectada\"\n}. Don't add ```json or ```, just the pure JSON. Don't add a dash in the plate number."},
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
            plate_number = plate_data.get("placa", "No Detectada")
            print(f"Placa detectada: {plate_number}")

            if plate_number != "No Detectada" and is_valid_plate(plate_number):
                write_to_csv(plate_number, image_filename)
                upload_to_server(plate_number, image_path)

            return plate_number
        except json.JSONDecodeError:
            print("Error al decodificar el JSON de la respuesta.")
            print(f"Respuesta recibida: {response_content}")
            return "No Detectada"
        
    except Exception as e:
        # Handle other possible errors
        print(f"An unexpected error occurred: {e}")
