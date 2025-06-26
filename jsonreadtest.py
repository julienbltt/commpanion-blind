import cv2
from paddleocr import PaddleOCR
import json
import pyttsx3
import os  

# === Configuration ===
CAMERA_INDEX = 2 # Try 1 or 2 if this doesn't work
CAPTURED_IMAGE_PATH = "captured_from_camera.jpg"
OCR_OUTPUT_DIR = "output"

# === Step 1: Capture frame from external camera ===
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

print("📸 Capturing image...")
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to capture frame.")
    exit()

# Optional: Resize the image if needed
scale_percent = 100
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
resized_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

cv2.imwrite(CAPTURED_IMAGE_PATH, resized_img)
print(f"✅ Image saved to {CAPTURED_IMAGE_PATH}")

# === Step 2: Run OCR ===
ocr = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True
)

result = ocr.predict(CAPTURED_IMAGE_PATH)

for res in result:
    res.print()
    res.save_to_img(OCR_OUTPUT_DIR)
    res.save_to_json(OCR_OUTPUT_DIR)

# === Step 3: Read JSON Result and Speak ===
json_name = os.path.basename(CAPTURED_IMAGE_PATH).split('.')[0] + "_res.json"
json_path = os.path.join(OCR_OUTPUT_DIR, json_name)

if not os.path.exists(json_path):
    print(f"❌ OCR result not found at {json_path}")
    exit()

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


rec_scores = data.get("rec_scores", [])
if len(rec_scores)==0:
    print("No Text Detected")

else:
    avg = sum(rec_scores)/len(rec_scores)
    if avg<.90:
        error = "Low Confidence, retake photo"
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        print(error,avg)
        engine.say(error)
        engine.runAndWait()
    else:
        rec_texts = data.get("rec_texts", [])
        full_text = ', '.join(rec_texts).strip()

        print("📝 Extracted Text:",avg)
        print(full_text)

        # === Step 4: Use TTS to speak the text ===
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)

        engine.say(full_text)
        engine.runAndWait()
