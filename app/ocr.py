import pytesseract
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load TrOCR model once at startup
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")  # or your fine-tuned model path
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def process_image(image: Image.Image) -> str:
    # Get word-level bounding boxes from pytesseract
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60 and data['text'][i].strip() != "":
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            word_img = image.crop((x, y, x + w, y + h))

            # Resize to reduce memory usage
            word_img = word_img.resize((224, 224))

            # Process and infer
            pixel_values = processor(images=word_img, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            words.append(generated_text)

    return " ".join(words)
