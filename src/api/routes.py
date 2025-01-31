from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import Dict

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
connections: Dict[str, WebSocket] = {}

@app.post("/translate")
async def translate_audio(file: UploadFile = File(...)):
    """Endpoint for file-based translation"""
    audio_content = await file.read()
    # Process audio content through the pipeline
    result = translation_pipeline.process(audio_content)
    return {
        "source_text": result["source_text"],
        "translated_text": result["translated_text"]
    }

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint for real-time streaming translation"""
    await websocket.accept()
    
    try:
        while True:
            # Receive audio chunks
            audio_data = await websocket.receive_bytes()
            
            # Process through pipeline
            result = translation_pipeline.process(audio_data)
            
            # Send back results
            await websocket.send_json({
                "source_text": result["source_text"],
                "translated_text": result["translated_text"]
            })
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
