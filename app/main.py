from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.ocr import process_image
from PIL import Image
import io

app = FastAPI()

@app.post("/predict/")
async def predict_text(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        text = process_image(image)
        return JSONResponse(content={"text": text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
