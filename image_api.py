import base64
from mimetypes import guess_type
import os
from dotenv import load_dotenv
from openai import OpenAI

 # Load environment variables from a .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI()

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

def print_license_plate():

    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the image file name
    image_filename = 'car.png'
    
    # Construct the full path to the image
    image_path = os.path.join(script_dir, image_filename)
    
    try:
        # Convert the local image to a Data URL
        data_url = local_image_to_data_url(image_path)
        # print("Data URL successfully created.")
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
                        {"type": "text", "text": "Extract and return the license plate number of the vehicle visible in the provided image, formatted as a JSON object."},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ],
                }
            ],
            max_tokens=300,
        )
       
       print(response.choices[0].message.content)
        
    except Exception as e:
        # Handle other possible errors
        print(f"An unexpected error occurred: {e}")