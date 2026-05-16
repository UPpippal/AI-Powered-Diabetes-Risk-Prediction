import google.generativeai as genai
from dotenv import load_dotenv
import google.api_core.exceptions
import os

# Load .env
load_dotenv()

# Configure API
genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Gemini model
model = genai.GenerativeModel(
    "models/gemini-2.0-flash"
)


def get_health_advice(user_query):

    prompt = f"""
    You are a diabetes health assistant.

    Give short, safe and useful health advice.

    User question:
    {user_query}

    Keep answer simple.
    Do not diagnose disease.
    """

    try:
        response = model.generate_content(prompt)
        return response.text

    except google.api_core.exceptions.ResourceExhausted:

        return """
### Personalized Health Recommendations

✅ Exercise regularly (30 min daily)

✅ Reduce sugar intake

✅ Maintain healthy BMI

✅ Eat fruits and vegetables

✅ Monitor blood pressure & cholesterol

✅ Stay hydrated and sleep properly

⚠️ Please consult a doctor for medical advice.
"""