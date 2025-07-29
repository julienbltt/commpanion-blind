import cv2
import torch
import logging
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import os


import torch.backends.mkldnn
torch.set_num_threads(10)  # Use the number of physical CPU cores you have
torch.backends.mkldnn.enabled = True

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

def load_instructblip_model():
    logger.info("Loading InstructBLIP model (Salesforce/instructblip-flan-t5-xl)...")
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", use_fast=True)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device =  "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model running on: {device}")
    return processor, model, device

def generate_caption(image_path, processor, model, device):
    try:
        image = Image.open(image_path).convert("RGB")
        prompt = "What's in this image."

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                num_beams=1
            )

        caption = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return "Caption generation failed."
    
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Camera not accessible.")
        return None
    
    
    #print("Press SPACE to capture the image, or ESC to cancel.")
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to capture image from camera.")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    img_path = "captured.jpg" # Changed filename to avoid overwriting "captured.jpg"
    cv2.imwrite(img_path, frame)
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Image captured automatically and saved as {img_path}")
    return img_path

# Add a setup function to initialize once
def initialize_blip():
    return load_instructblip_model()