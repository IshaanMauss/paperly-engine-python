from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Sirf vision models print karenge
models = client.models.list()
for model in models.data:
    if "vision" in model.id:
        print(f"✅ EXACT MODEL ID: {model.id}")