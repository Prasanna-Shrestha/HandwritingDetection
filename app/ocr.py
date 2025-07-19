import pytesseract
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load TrOCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("popPrasanna/trocr_handwritten")
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def process_image(image: Image.Image) -> str:
    # Word-level OCR bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 50 and data['text'][i].strip() != "":
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            word_img = image.crop((x, y, x + w, y + h))

            pixel_values = processor(images=word_img, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            words.append(generated_text)
    
    return " ".join(words)
