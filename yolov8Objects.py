#YOLO WITH BLIP
import cv2
import torch
import logging
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from ultralytics import YOLO
import lmstudio as lms


#-----------------------LLM Object Locator------------------------
class ObjectLocator:
    def __init__(self, model_name="mistral-7b-instruct-v0.3"):
        self.model = lms.llm(model_name)
        self.chat = lms.Chat()

    def locate(self, prompt: str, locations: str) -> str:
        instructions = (
            "You will be given a list of sentences and a question. "
            "Return every sentence from the list that answers the question. "
            "Do not add explanations. Do not add extra words. "
            "Just return the matching sentences from the list."
            "If there is no matching sentence say that the object is not present"
        )

        full_prompt = f"{instructions}\n\nSentences:\n{locations}\n\nQuestion: {prompt}\nAnswer:"
        self.chat.add_user_message(full_prompt)

        chunks = self.model.respond_stream(
            self.chat,
            on_message=self.chat.append,
        )

        return "".join(chunk.content for chunk in chunks).strip()
# Set performance settings
torch.set_num_threads(10)
torch.backends.mkldnn.enabled = True

# Logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Camera not accessible.")
        return None
    
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to capture image from camera.")
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    img_path = "captured.jpg"
    cv2.imwrite(img_path, frame)
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Image captured automatically and saved as {img_path}")
    return img_path

# Load YOLOv8
def load_yolo_model():
    logger.info("Loading YOLOv8...")
    return YOLO('yolov8l-oiv7.pt')  # change to 'yolov8s-oiv7.pt' or 'yolov8m-oiv7.pt' if desired (for less processing power)


# YOLO detection
"""def detect_objects(image_path, yolo_model):
    results = yolo_model(image_path)
    detected = []
    for r in results:
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf > 0.4:
                label = yolo_model.names[int(box.cls[0].item())]
                detected.append(label)
    return detected"""

# # InstructBLIP caption generation
# def generate_caption(image_path, prompt, processor, model, device):
#     try:
#         image = Image.open(image_path).convert("RGB")
#         inputs = processor(images=image, text=prompt, return_tensors="pt")
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=50,
#                 do_sample=False,
#                 num_beams=1
#             )
#         return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
#     except Exception as e:
#         logger.error(f"Caption generation failed: {e}")
#         return "Caption generation failed."


def describe_all_objects(image_path, yolo_model):
    results = yolo_model(image_path)
    descriptions = []
    for r in results:
        width = r.orig_shape[1]
        height = r.orig_shape[0]
        for box in r.boxes:
            conf = box.conf[0].item()
            if conf < 0.4:
                continue

            label_idx = int(box.cls[0].item())
            label = yolo_model.names[label_idx]

            x_center = (box.xywh[0][0]).item()
            y_center = (box.xywh[0][1]).item()

            # Horizontal segmentation (9 regions)
            h_segment = width / 9
            if x_center < h_segment:
                horiz = "far left"
            elif x_center < 2 * h_segment:
                horiz = "left"
            elif x_center < 3 * h_segment:
                horiz = "close left"
            elif x_center < 4 * h_segment:
                horiz = "slightly left"
            elif x_center < 5 * h_segment:
                horiz = "middle"
            elif x_center < 6 * h_segment:
                horiz = "slightly right"
            elif x_center < 7 * h_segment:
                horiz = "close right"
            elif x_center < 8 * h_segment:
                horiz = "right"
            else:
                horiz = "far right"

            # Vertical segmentation (3 regions)
            v_segment = height / 3
            if y_center < v_segment:
                vert = "far away"
            elif y_center < 2 * v_segment:
                vert = "mid range"
            else:
                vert = "close up"

            descriptions.append(f"There is a {label} {horiz}, and {vert}.")

    if not descriptions:
        return "No objects found in the image."
    return "\n".join(descriptions)

logger = setup_logging()
# Main loop

def mainObjectLocator(prompt):
    yolo_model = load_yolo_model()
    locator=ObjectLocator()

    #print("\nPress SPACE to capture image, or ESC to quit.")
    image_path = capture_image()
    if image_path:
        logger.info("Detecting all objects in image using YOLO...")
        locations = describe_all_objects(image_path, yolo_model)
        print(locations)
        
        #prompt = "Where is the laptop"
        result=locator.locate(prompt, locations)
        print("Results",result)
        return result
    else:
        print("I couldn't capture the image")
        return "I couldn't capture the image"