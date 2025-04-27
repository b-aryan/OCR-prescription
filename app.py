from flask import Flask, render_template, request, jsonify
import pytesseract
from PIL import Image
import json
from openai import OpenAI

app = Flask(__name__)

# Your OpenAI API key


# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def extract_text_from_image(image):
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(image)
    return text

def process_text(text):


    # JSON format example
    json_format_example = {
        "patient_name": "John Doe",
        "medicines": [
            {
                "medicine_name": "Aspirin",
                "dosage": "100mg"
            },
            {
                "medicine_name": "Paracetamol",
                "dosage": "500mg"
            }
        ]
    }

    # Convert JSON format to string
    json_format_str = json.dumps(json_format_example, indent=2)
    prompt_message_system = "You are a helpful assistant understanding the skewed OCR extracted text from a medical prescription. However, the text is filled with noise and spelling mistakes due to scanned copies. Your task is to correct the noise, mistakes and extract only the name of the patient and prescribed medicines with dosages. Omit any extraneous information. In case of incorrect medicine names, I will utilize my knowledge of medicine to identify the closest match and corresponding dosages.\n\n Please provide the text from the medical prescription.\n\n JSON Format Example:\n" + json_format_str

    # Send the prompt to OpenAI
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt_message_system},
            {"role": "user", "content": text}
        ]
    )

    processed_text = response.choices[0].message.content
    return processed_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_text', methods=['POST'])
def extract_text():
    image = request.files['image']
    image.save('uploaded_image.jpg')
    image_data = Image.open(image)
    text = extract_text_from_image(image_data)
    processed_text = process_text(text)

    return jsonify({'text': processed_text})

if __name__ == '__main__':
    app.run(debug=True)