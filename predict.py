from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import numpy as np
import io
import os
import pytesseract

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pytesseract.pytesseract.tesseract_cmd = os.path.join(BASE_DIR, "bin", "tesseract")
os.environ["TESSDATA_PREFIX"] = os.path.join(BASE_DIR, "bin")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load TrOCR model and processor once
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("popPrasanna/trocr_handwritten").to(DEVICE)
model.eval()

def preprocess_image(image_bytes):
    np_buf = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if image_cv is None:
        raise ValueError("Could not decode image")
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return image_rgb

def predict_from_image(image_bytes):
    image_rgb = preprocess_image(image_bytes)
    
    # Simple: do full image inference for now
    pil_img = Image.fromarray(image_rgb)
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=256)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()
