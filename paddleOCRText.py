import cv2
from paddleocr import PaddleOCR
import json
import os

# === Configuration ===
# These paths are still used for saving the temporary image and OCR output files
CAPTURED_IMAGE_PATH = "captured_from_camera.jpg"
OCR_OUTPUT_DIR = "output"

# Initialize PaddleOCR globally, so it's loaded only once
print("⏳ Initializing PaddleOCR model... This might take a moment (first run only).")
ocr = PaddleOCR(
    lang='en',
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True
)
print("✅ PaddleOCR model initialized.")

def paddleText(frame):
    """
    Performs OCR on a given image frame using the globally initialized PaddleOCR object.
    It saves the frame to a temporary file, runs OCR, processes results,
    and returns the extracted text or a status message.

    Args:
        frame (numpy.ndarray): The image frame (from CameraManager) to perform OCR on.

    Returns:
        str: The extracted text, "No Text Detected", "Low confidence. Please retake the photo.",
             "Error saving image for OCR.", or "Error during OCR processing".
    """
    if frame is None:
        print("❌ No frame provided for OCR.")
        return "Error: No image frame available for OCR."

    # Save the captured frame to a file
    try:
        cv2.imwrite(CAPTURED_IMAGE_PATH, frame)
        print(f"✅ Image saved to {CAPTURED_IMAGE_PATH}")
    except Exception as e:
        print(f"❌ Error saving image for OCR: {e}")
        return "Error: Could not save image for OCR."

    # === Step 2: Run OCR ===
    try:
        # Using ocr.ocr directly
        result = ocr.ocr(CAPTURED_IMAGE_PATH, cls=True)
    except Exception as e:
        print(f"❌ Error during OCR processing: {e}")
        return "Error: OCR processing failed."

    # Check if any text was detected
    if not result or not result[0]:
        return "No Text Detected"

    # Process the OCR result directly in memory
    rec_texts = []
    rec_scores = []
    for line in result[0]:
        if line and len(line) > 1 and len(line[1]) > 1:
            text = line[1][0]
            confidence = line[1][1]
            rec_texts.append(text)
            rec_scores.append(confidence)

    # Optional: Save to JSON for debugging/review (can be commented out for production)
    json_name = os.path.basename(CAPTURED_IMAGE_PATH).split('.')[0] + "_res.json"
    json_path = os.path.join(OCR_OUTPUT_DIR, json_name)
    os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)
    processed_result = {
        "rec_texts": rec_texts,
        "rec_scores": rec_scores
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_result, f, ensure_ascii=False, indent=4)
    print(f"✅ OCR result details saved to {json_path}")


    # Calculate average confidence
    if not rec_scores:
        return "No Text Detected"

    avg_confidence = sum(rec_scores) / len(rec_scores)
    print(f"📝 Extracted Text Average Confidence: {avg_confidence:.2f}")

    # Decision based on confidence threshold
    if avg_confidence < 0.9: # Adjust this threshold as needed
        return "Low confidence. Please retake the photo."
    else:
        full_text = ' '.join(rec_texts).strip()
        if not full_text:
            return "No Text Detected"
        print(f"📖 Extracted Text: {full_text}")
        return full_text