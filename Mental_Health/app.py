import os
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai

app = Flask(__name__)

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define the generation parameters
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="This need to welcome each and every user once they come first and need to understand their symptom \nand show the respective therapist name and diseases that affects them. This need to show only two things in output\none thing is the name of the disease, then the name of the specialist like cardiologist,general oractitioner . These are the only two outputs expected. \nThe answer needs to be short and precise, not too long, and ends with a greeting l. \nIt should not talk other than this to the user. If you need to ask questions to get clarity about the patient, \nyou can ask only two questions, not more than that. And that question also needs to be very small and precise, \nwhich makes it easy for the user to understand and answer, not using more medical terms while asking questions to the user.this should not give the names of the doctor like DR.Jones,this should only suggest the threpist only",
)

# Initialize chat session
chat_session = model.start_chat(
    history=[
        {"role": "user", "parts": ["hi\n"]},
        {"role": "model", "parts": ["What is your main concern?  \n \nOnce you tell me, I can suggest a therapist.  \n\nGet well soon. \n"]},
        {"role": "user", "parts": ["i have sever head pain"]},
        {"role": "model", "parts": ["Migraine, General Practioner \n\nTake care! \n"]},
        {"role": "user", "parts": ["i have cough and sore throat"]},
        {"role": "model", "parts": ["Common Cold, General Practioner \n\nFeel better soon! \n"]},
        {"role": "user", "parts": ["i have cough"]},
        {"role": "model", "parts": ["Can you tell me if you also have a fever? \n\n"]},
        {"role": "user", "parts": ["yes"]},
        {"role": "model", "parts": ["Bronchitis, General Practioner\n\nTake care! \n"]},
        {"role": "user", "parts": ["i have head ace"]},
        {"role": "model", "parts": ["Tension Headache,  General Practitioner\n\nHope you feel better soon! \n"]},
        {"role": "user", "parts": ["i have sever back pain"]},
        {"role": "model", "parts": ["Back Strain, Physiotherapist\n\nGet well soon! \n"]},
    ]
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")

        if user_input is None:
            return jsonify({"error": "Invalid request"}), 400

        print(f"Received user input: {user_input}")  # Log user input to the terminal

        # Generate the response from the model
        response = chat_session.send_message(user_input)

        model_response = response.text

        print(f"Model response: {model_response}")  # Log model response

        return jsonify({"response": model_response})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/")
def index():
    return send_from_directory('static', 'index.html')

if __name__ == "__main__":
    app.run(debug=True)
