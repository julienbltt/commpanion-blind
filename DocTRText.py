import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
import os
import logging

# === Configuration ===
# These paths are still used for saving the temporary image and OCR output files
CAPTURED_IMAGE_PATH = "captured_from_camera.jpg"
OCR_OUTPUT_DIR = "output"

# Initialize PaddleOCR globally, so it's loaded only once
print("⏳ Initializing DocTR model... This might take a moment (first run only).")
model = ocr_predictor(pretrained=True)
print("✅ DocTR model initialized.")
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

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
    
    img_path = "captured_auto.jpg" # Changed filename to avoid overwriting "captured.jpg"
    cv2.imwrite(img_path, frame)
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Image captured automatically and saved as {img_path}")
    return frame

def DocTRRead(frame):
    """
    Performs OCR on a given image frame using the globally initialized OCR object.
    It saves the frame to a temporary file, runs OCR, processes results,
    and returns the extracted text or a status message.

    Args:
        frame (numpy.ndarray): The image frame (from CameraManager) to perform OCR on.

    Returns:
        str: The extracted text, "No Text Detected", "Low confidence. Please retake the photo.",
             "Error saving image for OCR.", or "Error during OCR processing".
    """
    if frame is None:
        print("No frame provided for OCR.")
        return "Error: No image frame available for OCR."

    # Save the captured frame to a file
    try:
        cv2.imwrite(CAPTURED_IMAGE_PATH, frame)
        print(f"Image saved to {CAPTURED_IMAGE_PATH}")
    except Exception as e:
        print(f"Error saving image for OCR: {e}")
        return "Error: Could not save image for OCR."

    # === Step 2: Run OCR ===
    try:
        # Using ocr.ocr directly
        doc = DocumentFile.from_images(CAPTURED_IMAGE_PATH)
        result = model(doc)
        output = result.export()
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return "Error: OCR processing failed."

    # Check if any text was detected
    if not result:
        return "No Text Detected"
    json_name = os.path.basename(CAPTURED_IMAGE_PATH).split('.')[0] + "_res.json"
    json_path = os.path.join(OCR_OUTPUT_DIR, json_name)
    os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
    with open(json_path, "w") as f:
        f.write(json.dumps(output, indent=1))
    f.close()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extraire tous les mots
    words = []
    for page in data.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    value = word.get("value")
                    if value:
                        words.append(value)

    # Concaténer les mots en une seule chaîne
    result = " ".join(words)
    print(result)
    return result
