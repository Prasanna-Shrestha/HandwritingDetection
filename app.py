import gradio as gr
from predict import predict_from_image

def ocr_interface(image):
    # image is a PIL Image from Gradio by default
    import io
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return predict_from_image(byte_im)

iface = gr.Interface(fn=ocr_interface, inputs=gr.Image(type="pil"), outputs="text")
iface.launch()
