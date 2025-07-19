# predict.py (refactored for serverless / no local writes)
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, cv2, pytesseract, os, io, numpy as np

# Optional: pick device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Model loading (do this once) ----
# Prefer packaging the model inside the deployment image / layer.
MODEL_PATH = os.getenv("MODEL_PATH", "model/checkpoint-5080")

# If you keep using a HF public model instead of checkpoint, you could:
# MODEL_ID = "microsoft/trocr-base-handwritten"
# processor = TrOCRProcessor.from_pretrained(MODEL_ID)
# model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("popPrasanna/trocr_handwritten")
# model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

def predict_from_image(image_bytes: bytes) -> str:
    # Read image once
    np_buf = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if image_cv is None:
        raise ValueError("Could not decode image")

    # Convert BGR->RGB for PIL/Tesseract consistency
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # --- Word detection via Tesseract (if available) ---
    # NOTE: On Vercel you probably *won't* have the tesseract binary (see section 3).
    data = pytesseract.image_to_data(
        image_rgb,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    words = []
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        # Filter tiny boxes
        if w < 20 or h < 20:
            continue
        crop = image_rgb[y:y+h, x:x+w]
        pil_crop = Image.fromarray(crop)

        pixel_values = processor(images=pil_crop, return_tensors="pt").pixel_values.to(DEVICE)
        # Quick sanity check
        if pixel_values.ndim != 4 or pixel_values.shape[-1] < 10 or pixel_values.shape[-2] < 10:
            continue

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=32)
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if pred:
            words.append((y, x, pred))  # keep coords for ordering

    # If no words extracted (maybe Tesseract absent), fallback: run TrOCR on entire image
    if not words:
        pil_full = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = processor(images=pil_full, return_tensors="pt").pixel_values.to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=256)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Sort words by line (y) then x
    words.sort(key=lambda r: (r[0]//40, r[1]))
    sentence = " ".join(w[2] for w in words)
    return sentence
