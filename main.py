# File: main.py
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from api.extract_router import router as extract_router

# Load environment variables
load_dotenv()

app = FastAPI(title="Paperly AI Engine", version="1.0")

# Security: Allow Node.js to communicate with this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("NODE_BACKEND_ORIGIN", "http://localhost:5000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_router, prefix="/api")

@app.get("/")
def health_check():
    return {"status": "success", "message": "Python Intelligence Engine is running! 🧠"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)