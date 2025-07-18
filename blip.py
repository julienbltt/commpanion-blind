import os
import cv2
import time
import torch
import threading
import logging
import numpy as np
from PIL import Image
from contextlib import contextmanager
from typing import Optional, Tuple, Union
from transformers import BlipProcessor, BlipForConditionalGeneration


class BlipModel:
    """Thread-safe BLIP model for image captioning with camera capture capabilities"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        image_dir: str = "image",
        delete_after_caption: bool = True,
        camera_id: int = 0
    ):
        self.model_name = model_name
        self.image_dir = image_dir
        self.delete_after_caption = delete_after_caption
        self.camera_id = camera_id

        os.makedirs(self.image_dir, exist_ok=True)

        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_lock = threading.Lock()
        self._loaded = False

        self.logger = logging.getLogger("BlipModel")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def load(self):
        """Load the BLIP model"""
        self.logger.info("🔄 Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self._loaded = True
        self.logger.info("✅ BLIP model loaded.")

    def generate_caption(self, image: Union[str, Image.Image, np.ndarray],
                         max_length: int = 50,
                         num_beams: int = 5,
                         early_stopping: bool = True,
                         temperature: float = 1.0) -> str:
        """Generate a caption from an image."""
        temp_image_path = None
        caption = "Caption generation failed."

        if isinstance(image, (Image.Image, np.ndarray)):
            pil_image = self._convert_to_pil(image)
            os.makedirs("temp_images", exist_ok=True)
            temp_image_path = os.path.join("temp_images", f"temp_{int(time.time() * 1000)}.jpg")
            pil_image.save(temp_image_path)
            image_path = temp_image_path
        elif isinstance(image, str):
            image_path = image
            pil_image = self._convert_to_pil(image)
        else:
            raise ValueError("Unsupported image input type.")

        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            caption_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                temperature=temperature,
                do_sample=False
            )

        caption = self.processor.tokenizer.decode(caption_ids[0], skip_special_tokens=True)
        self.logger.info(f"Generated caption: {caption}")

        if temp_image_path:
            self._delete_image(temp_image_path)

        return caption

    def _convert_to_pil(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """Convert any supported image format to a PIL image."""
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            return Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    @contextmanager
    def _get_camera(self, camera_id: int = 0):
        """Safe camera access with context manager"""
        cap = cv2.VideoCapture(camera_id)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Cannot access camera {camera_id}")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            yield cap
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def capture_image_auto(self, output_path: str = "captured/captured_auto.jpg",
                           camera_id: int = 0,
                           stabilization_time: float = 0.1,
                           warmup_frames: int = 1) -> Optional[str]:
        """Capture a single image automatically from the webcam"""
        with self._get_camera(camera_id) as cap:
            self.logger.info("Initializing camera for auto capture...")
            for _ in range(warmup_frames):
                ret, _ = cap.read()
                if not ret:
                    self.logger.error("Failed to read frames during warmup")
                    return None
                time.sleep(0.1)

            time.sleep(stabilization_time)

            best_frame = None
            best_brightness = -1

            for _ in range(5):
                ret, frame = cap.read()
                if not ret:
                    continue
                brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if brightness > best_brightness:
                    best_brightness = brightness
                    best_frame = frame.copy()
                time.sleep(0.1)

            if best_frame is None:
                self.logger.error("Failed to capture any usable frame")
                return None

            cv2.imwrite(output_path, best_frame)
            self.logger.info(f"📸 Image captured: {output_path} (Brightness: {best_brightness:.1f})")
            return output_path

    def capture_and_describe(self,
                            auto_capture: bool = True,
                            output_path: Optional[str] = None,
                            camera_id: Optional[int] = None,
                            caption_params: Optional[dict] = None,
                            stabilization_time: float = 0.1) -> Tuple[Optional[str], Optional[str]]:
        """Capture an image and return a caption"""
        os.makedirs("image", exist_ok=True)
        timestamp = int(time.time() * 1000)
        image_path = output_path or os.path.join("image", f"temp_{timestamp}.jpg")

        if caption_params is None:
            caption_params = {}

        if camera_id is None:
            camera_id = self.camera_id  # Use the instance's camera ID

        image_result_path = self.capture_image_auto(image_path, camera_id, stabilization_time)

        if image_result_path is None:
            return None, None

        caption = self.generate_caption(image_result_path, **caption_params)

        if self.delete_after_caption:
            self._delete_image(image_result_path)

        return image_result_path, caption


    def _delete_image(self, image_path: str):
        """Remove the image file from disk if exists."""
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                self.logger.info(f"🗑️ Deleted: {image_path}")
        except Exception as e:
            self.logger.error(f"Error deleting image: {e}")

    def unload(self):
        """Unload model from memory."""
        with self._model_lock:
            if self._loaded:
                self.model = None
                self.processor = None
                self._loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("Model unloaded successfully")

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass


