from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from app.ocr import process_image

app = FastAPI()

# Enable CORS (important for mobile apps or web clients)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        text = process_image(image)
        return JSONResponse(content={"text": text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
