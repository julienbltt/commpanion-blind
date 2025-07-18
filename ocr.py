import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancementLevel(Enum):
    """Image enhancement levels for preprocessing."""
    LIGHT = "light"
    MEDIUM = "medium"
    STRONG = "strong"
    CUSTOM = "custom"


class PSMMode(Enum):
    """Page Segmentation Mode constants for Tesseract."""
    AUTO_OSD = 0
    AUTO_WITH_OSD = 1
    AUTO_WITHOUT_OSD = 2
    AUTO = 3
    SINGLE_COLUMN = 4
    SINGLE_BLOCK_VERT_TEXT = 5
    SINGLE_BLOCK = 6
    SINGLE_TEXT_LINE = 7
    SINGLE_WORD = 8
    CIRCLE_WORD = 9
    SINGLE_CHARACTER = 10
    SPARSE_TEXT = 11
    SPARSE_TEXT_OSD = 12
    RAW_LINE = 13


@dataclass
class OCRResult:
    """Data class to store OCR detection results."""
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    level: int


@dataclass
class ImageInfo:
    """Data class to store image information."""
    path: str
    size: Tuple[int, int]
    mode: str
    file_size_kb: float


@dataclass
class AnalysisStats:
    """Data class to store analysis statistics."""
    total_characters: int
    total_words: int
    detected_elements: int
    average_confidence: float


class TesseractOCR:
    """
    Professional Tesseract OCR wrapper with advanced preprocessing and analysis capabilities.
    
    This class provides a comprehensive interface for optical character recognition
    using Tesseract OCR engine with optimized preprocessing and detailed analysis features.
    """
    
    def __init__(self, tesseract_path: Optional[str] = None, language_cache: bool = True, camera_id: int = 0) -> None:
        """
        Initialize Tesseract OCR with automatic path detection and configuration.
        
        Args:
            tesseract_path: Optional custom path to tesseract executable
            language_cache: Enable language list caching for performance
            camera_id: Default camera ID for image capture
            
        Raises:
            Exception: If Tesseract installation is not found
        """
        self._configure_tesseract_path(tesseract_path)
        self._available_languages = self._get_available_languages() if language_cache else None
        self._image_cache: Dict[str, Image.Image] = {}
        self.camera_id = camera_id  
        
    def _configure_tesseract_path(self, custom_path: Optional[str]) -> None:
        """
        Configure Tesseract executable path with automatic detection.
        
        Args:
            custom_path: Optional custom path to tesseract executable
            
        Raises:
            Exception: If Tesseract is not found in any standard location
        """
        if custom_path:
            pytesseract.pytesseract.tesseract_cmd = custom_path
            return
            
        # Standard installation paths for Windows
        standard_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            'tesseract'  # PATH environment variable
        ]
        
        for path in standard_paths:
            try:
                pytesseract.pytesseract.tesseract_cmd = path
                pytesseract.get_tesseract_version()  # Test if working
                logger.info(f"Tesseract found at: {path}")
                return
            except Exception:
                continue
                
        raise Exception("Tesseract OCR not found. Please verify installation.")
    
    def _get_available_languages(self) -> List[str]:
        """
        Retrieve and cache available Tesseract languages.
        
        Returns:
            List of available language codes
        """
        try:
            languages = pytesseract.get_languages()
            logger.info(f"Available languages: {', '.join(languages)}")
            return languages
        except Exception as e:
            logger.warning(f"Failed to retrieve language list: {e}")
            return ['eng']  # Fallback to English
        
    def _deskew_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Correction automatique de l'inclinaison du texte.
        
        Args:
            img_array: Image en array NumPy
            
        Returns:
            Image corrigée
        """
        try:
            # Détection des contours pour trouver l'inclinaison
            edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calcul de l'angle moyen
                angles = []
                for rho, theta in lines[:10]:  # Prendre les 10 premières lignes
                    angle = theta * 180 / np.pi
                    # Normaliser l'angle
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                if angles:
                    avg_angle = np.mean(angles)
                    
                    # Correction seulement si l'inclinaison est significative
                    if abs(avg_angle) > 0.5:  # Plus de 0.5 degrés
                        height, width = img_array.shape
                        center = (width // 2, height // 2)
                        matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                        corrected = cv2.warpAffine(img_array, matrix, (width, height), 
                                                 flags=cv2.INTER_CUBIC, 
                                                 borderValue=255)
                        logger.debug(f"Correction d'inclinaison: {avg_angle:.2f}°")
                        return corrected
        
        except Exception as e:
            logger.debug(f"Correction d'inclinaison échouée: {e}")
        
        return img_array

    def preprocess_image_custom(self, 
                        image_input: Union[str, Path, Image.Image, np.ndarray],
                        target_height: int = 800,
                        apply_deskew: bool = True,
                        block_size: int = 11,
                        c_value: int = 2) -> Image.Image:
        """
        Preprocessing optimisé pour une meilleure reconnaissance OCR.
        
        Args:
            image_input: Image d'entrée
            target_height: Hauteur cible minimale
            apply_deskew: Correction de l'inclinaison
            block_size: Taille du bloc pour binarisation adaptative
            c_value: Constante pour binarisation adaptative
            
        Returns:
            Image préprocessée optimisée pour OCR
        """
        image = self._load_image(image_input)
        original_size = image.size
        
        # 1. Conversion en niveaux de gris
        if image.mode != 'L':
            image = image.convert('L')
        
        # 2. Redimensionnement intelligent
        if image.height < target_height:
            ratio = target_height / image.height
            new_size = (int(image.width * ratio), target_height)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Image redimensionnée de {original_size} vers {new_size}")
        
        # 3. Conversion en array NumPy pour les traitements OpenCV
        img_array = np.array(image)
        
        # 4. Détection et inversion automatique (texte blanc sur fond noir)
        mean_val = np.mean(img_array)
        if mean_val < 127:
            img_array = 255 - img_array
            logger.debug("Image inversée (texte blanc détecté)")
        
        # 5. Débruitage léger avec filtre gaussien
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        # 6. Binarisation adaptative (plus efficace que l'ajustement de contraste)
        binary = cv2.adaptiveThreshold(
            img_array, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,  # Taille du bloc pour le calcul du seuil
            C=c_value              # Constante soustraite de la moyenne
        )
        
        # 7. Nettoyage morphologique pour éliminer le bruit
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # 8. Correction de l'inclinaison (optionnel)
        if apply_deskew:
            cleaned = self._deskew_image(cleaned)
        
        return Image.fromarray(cleaned)
    
    @property
    def available_languages(self) -> List[str]:
        """Get list of available Tesseract languages."""
        if self._available_languages is None:
            self._available_languages = self._get_available_languages()
        return self._available_languages
    
    def _load_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load and convert various image input types to PIL Image.
        
        Args:
            image_input: Image file path, PIL Image, or numpy array
            
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If image input type is not supported
        """
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            if image_path in self._image_cache:
                return self._image_cache[image_path].copy()
            image = Image.open(image_path)
            self._image_cache[image_path] = image.copy()
            return image
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def preprocess_image(self, 
                        image_input: Union[str, Path, Image.Image, np.ndarray],
                        enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM,
                        min_size: int = 300) -> Image.Image:
        """
        Apply intelligent preprocessing to optimize image for OCR.
        
        This method applies various image enhancement techniques including
        resizing, contrast enhancement, sharpening, and noise reduction.
        
        Args:
            image_input: Input image in various formats
            enhancement_level: Level of preprocessing enhancement
            min_size: Minimum dimension for image resizing
            
        Returns:
            Preprocessed PIL Image optimized for OCR
        """
        image = self._load_image(image_input)
        original_size = image.size
        
        # Convert to grayscale for better OCR performance
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize if too small (OCR works better on larger images)
        if min(image.size) < min_size:
            ratio = min_size / min(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Image resized from {original_size} to {new_size}")
        
        # Apply enhancements based on level
        if enhancement_level in [EnhancementLevel.MEDIUM, EnhancementLevel.STRONG]:
            # Contrast enhancement
            contrast_factor = 1.5 if enhancement_level == EnhancementLevel.MEDIUM else 2.0
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
            
            # Sharpness enhancement
            sharpness_factor = 1.2 if enhancement_level == EnhancementLevel.MEDIUM else 1.5
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness_factor)
        
        if enhancement_level == EnhancementLevel.STRONG:
            # Noise reduction with median filter
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Adaptive thresholding for better contrast
            img_array = np.array(image)
            img_array = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            image = Image.fromarray(img_array)

        if enhancement_level == EnhancementLevel.CUSTOM:
            # Utiliser le preprocessing optimisé
            return self.preprocess_image_custom(image_input, min_size)
        
        return image
    
    def extract_text(self, 
                    image_input: Union[str, Path, Image.Image, np.ndarray],
                    languages: Union[str, List[str]] = 'eng',
                    psm_mode: PSMMode = PSMMode.SINGLE_BLOCK,
                    enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM,
                    custom_config: Optional[str] = None) -> str:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image_input: Input image in various formats
            languages: Language code(s) for OCR ('eng', 'fra', or ['eng', 'fra'])
            psm_mode: Page segmentation mode
            enhancement_level: Image preprocessing level
            custom_config: Custom Tesseract configuration string
            
        Returns:
            Extracted text as string
        """
        processed_image = self.preprocess_image(image_input, enhancement_level)
        
        # Handle language specification
        if isinstance(languages, list):
            lang_string = '+'.join(languages)
        else:
            lang_string = languages
        
        # Build configuration string
        if custom_config:
            config = custom_config
        else:
            config = f'--oem 3 --psm {psm_mode.value}'
        
        try:
            start_time = time.perf_counter()
            text = pytesseract.image_to_string(processed_image, lang=lang_string, config=config)
            processing_time = time.perf_counter() - start_time
            
            logger.debug(f"Text extraction completed in {processing_time:.3f}s")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def extract_text_with_boxes(self, 
                               image_input: Union[str, Path, Image.Image, np.ndarray],
                               languages: Union[str, List[str]] = 'eng',
                               confidence_threshold: float = 30.0,
                               enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM) -> List[OCRResult]:
        """
        Extract text with bounding box coordinates and confidence scores.
        
        Args:
            image_input: Input image in various formats
            languages: Language code(s) for OCR
            confidence_threshold: Minimum confidence score (0-100)
            enhancement_level: Image preprocessing level
            
        Returns:
            List of OCRResult objects containing text, coordinates, and confidence
        """
        processed_image = self.preprocess_image(image_input, enhancement_level)
        
        # Handle language specification
        if isinstance(languages, list):
            lang_string = '+'.join(languages)
        else:
            lang_string = languages
        
        try:
            data = pytesseract.image_to_data(
                processed_image,
                lang=lang_string,
                config='--oem 3 --psm 6',
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            for i in range(len(data['text'])):
                confidence = float(data['conf'][i])
                text = data['text'][i].strip()
                
                # Filter by confidence and non-empty text
                if confidence >= confidence_threshold and text:
                    result = OCRResult(
                        text=text,
                        confidence=confidence,
                        x=int(data['left'][i]),
                        y=int(data['top'][i]),
                        width=int(data['width'][i]),
                        height=int(data['height'][i]),
                        level=int(data['level'][i])
                    )
                    results.append(result)
            
            logger.info(f"Detected {len(results)} elements with confidence >= {confidence_threshold}%")
            return results
            
        except Exception as e:
            logger.error(f"Box extraction failed: {e}")
            return []
    
    def draw_detection_boxes(self, 
                           image_path: Union[str, Path],
                           results: List[OCRResult],
                           output_path: Union[str, Path] = "ocr_results.png",
                           show_confidence: bool = True,
                           min_confidence: float = 50.0,
                           box_colors: Optional[Dict[str, Tuple[int, int, int]]] = None) -> bool:
        """
        Draw OCR detection results on image with colored bounding boxes.
        
        Args:
            image_path: Path to original image
            results: List of OCRResult objects
            output_path: Output image path
            show_confidence: Include confidence scores in labels
            min_confidence: Minimum confidence for display
            box_colors: Custom colors for different confidence levels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Default color scheme based on confidence
            if box_colors is None:
                box_colors = {
                    'high': (0, 255, 0),    # Green for >80%
                    'medium': (0, 255, 255), # Yellow for 60-80%
                    'low': (0, 165, 255)     # Orange for <60%
                }
            
            drawn_count = 0
            for result in results:
                if result.confidence < min_confidence:
                    continue
                
                # Determine color based on confidence
                if result.confidence >= 80:
                    color = box_colors['high']
                elif result.confidence >= 60:
                    color = box_colors['medium']
                else:
                    color = box_colors['low']
                
                # Draw bounding box
                cv2.rectangle(image, 
                            (result.x, result.y), 
                            (result.x + result.width, result.y + result.height), 
                            color, 2)
                
                # Prepare label
                if show_confidence:
                    label = f"{result.text} ({result.confidence:.1f}%)"
                else:
                    label = result.text
                
                # Truncate long labels
                if len(label) > 30:
                    label = label[:27] + "..."
                
                # Draw label background and text
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(image, 
                            (result.x, result.y - 25), 
                            (result.x + label_size[0], result.y), 
                            color, -1)
                cv2.putText(image, label, (result.x, result.y - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                drawn_count += 1
            
            cv2.imwrite(str(output_path), image)
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Displayed {drawn_count}/{len(results)} boxes (confidence >= {min_confidence}%)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to draw detection boxes: {e}")
            return False
    
    def analyze_image(self, 
                     image_path: Union[str, Path],
                     languages: List[str] = None,
                     save_visualizations: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive OCR analysis on an image.
        
        Args:
            image_path: Path to image file
            languages: List of languages to test (defaults to available languages)
            save_visualizations: Save result visualizations
            
        Returns:
            Comprehensive analysis report dictionary
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return {}
        
        logger.info(f"Starting comprehensive analysis: {image_path.name}")
        
        # Get image information
        try:
            image = Image.open(image_path)
            file_size_kb = image_path.stat().st_size / 1024
            
            image_info = ImageInfo(
                path=str(image_path),
                size=image.size,
                mode=image.mode,
                file_size_kb=file_size_kb
            )
            
            logger.info(f"Image: {image_info.size[0]}x{image_info.size[1]}px, "
                       f"{image_info.mode}, {file_size_kb:.1f}KB")
            
        except Exception as e:
            logger.error(f"Failed to read image info: {e}")
            return {}
        
        # Use provided languages or available ones
        if languages is None:
            languages = [lang for lang in self.available_languages if lang in ['eng', 'fra', 'deu', 'spa']]
        
        # Filter unavailable languages
        valid_languages = [lang for lang in languages if lang in self.available_languages]
        if not valid_languages:
            logger.warning("No valid languages found, using English")
            valid_languages = ['eng']
        
        # Analysis results
        analysis_report = {
            'image_info': image_info,
            'results': {},
            'timestamp': time.time()
        }
        
        for lang in valid_languages:
            logger.info(f"Processing language: {lang}")
            
            try:
                # Extract text and boxes
                text = self.extract_text(image_path, languages=lang)
                boxes = self.extract_text_with_boxes(image_path, languages=lang)
                
                # Calculate statistics
                stats = AnalysisStats(
                    total_characters=len(text),
                    total_words=len(text.split()) if text else 0,
                    detected_elements=len(boxes),
                    average_confidence=np.mean([box.confidence for box in boxes]) if boxes else 0.0
                )
                
                analysis_report['results'][lang] = {
                    'text': text,
                    'boxes': [box.__dict__ for box in boxes],  # Convert to dict for serialization
                    'stats': stats.__dict__
                }
                
                logger.info(f"Language {lang}: {stats.total_words} words, "
                           f"{stats.detected_elements} elements, "
                           f"avg confidence: {stats.average_confidence:.1f}%")
                
                # Save visualization if requested
                if save_visualizations and boxes:
                    output_path = f"analysis_{lang}_{image_path.stem}.png"
                    self.draw_detection_boxes(image_path, boxes, output_path)
                
            except Exception as e:
                logger.error(f"Analysis failed for language {lang}: {e}")
                analysis_report['results'][lang] = {'error': str(e)}
        
        logger.info("Analysis completed successfully")
        return analysis_report
    
    def benchmark_psm_modes(self, image_path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
        """
        Benchmark different Page Segmentation Modes to find optimal settings.
        
        Args:
            image_path: Path to test image
            
        Returns:
            Dictionary with PSM mode results and performance metrics
        """
        logger.info(f"Benchmarking PSM modes for: {Path(image_path).name}")
        
        # Test modes suitable for different text layouts
        test_modes = [
            PSMMode.SINGLE_BLOCK,
            PSMMode.SINGLE_TEXT_LINE,
            PSMMode.SINGLE_WORD,
            PSMMode.SPARSE_TEXT,
            PSMMode.SPARSE_TEXT_OSD,
            PSMMode.RAW_LINE
        ]
        
        results = {}
        
        for psm_mode in test_modes:
            logger.info(f"Testing PSM {psm_mode.value}: {psm_mode.name}")
            
            try:
                start_time = time.perf_counter()
                text = self.extract_text(image_path, psm_mode=psm_mode)
                processing_time = time.perf_counter() - start_time
                
                word_count = len(text.split()) if text else 0
                char_count = len(text) if text else 0
                success = bool(text.strip())
                
                results[psm_mode.value] = {
                    'mode_name': psm_mode.name,
                    'text': text,
                    'word_count': word_count,
                    'char_count': char_count,
                    'processing_time': processing_time,
                    'success': success
                }
                
                status = "SUCCESS" if success else "FAILED"
                logger.info(f"PSM {psm_mode.value}: {status} - {word_count} words, "
                           f"{char_count} chars, {processing_time:.3f}s")
                
            except Exception as e:
                logger.error(f"PSM {psm_mode.value} failed: {e}")
                results[psm_mode.value] = {
                    'mode_name': psm_mode.name,
                    'error': str(e),
                    'success': False
                }
        
        # Find best performing mode
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if successful_results:
            best_mode = max(successful_results.keys(), 
                          key=lambda x: successful_results[x]['word_count'])
            logger.info(f"Recommended PSM mode: {best_mode} ({PSMMode(best_mode).name}) - "
                       f"{successful_results[best_mode]['word_count']} words detected")
        else:
            logger.warning("No PSM mode produced satisfactory results")
        
        return results
    
    def clear_cache(self) -> None:
        """Clear internal image cache to free memory."""
        self._image_cache.clear()
        logger.debug("Image cache cleared")

    def capture_image_from_webcam(self,
                               output_path: str = "image/captured_ocr.jpg",
                               warmup_frames: int = 5,
                               stabilization_time: float = 0.1) -> Optional[str]:
        """
        Capture an image from the webcam for OCR processing.

        Args:
            output_path: File path to save the captured image.
            warmup_frames: Number of initial frames to discard.
            stabilization_time: Delay after warmup for better image.

        Returns:
            Path to the saved image or None if failed.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        camera_id = self.camera_id  
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return None

        try:
            for _ in range(warmup_frames):
                cap.read()
                time.sleep(stabilization_time)

            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                return None

            cv2.imwrite(output_path, frame)
            logger.info(f"📸 Image captured to: {output_path}")
            return output_path

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def capture_and_extract_text(self,
                                camera_id: int = 0,
                                output_path: str = "image/captured_ocr.jpg",
                                languages: Union[str, List[str]] = 'eng',
                                psm_mode: PSMMode = PSMMode.SINGLE_BLOCK,
                                enhancement_level: EnhancementLevel = EnhancementLevel.MEDIUM) -> Optional[str]:
        """
        Capture image from webcam and extract OCR text.
        Automatically deletes the captured image after OCR.

        Args:
            camera_id: Webcam device ID.
            output_path: Path to temporarily save captured image.
            languages: OCR language(s).
            psm_mode: Page segmentation mode.
            enhancement_level: Image enhancement level before OCR.

        Returns:
            Extracted text or None if failed.
        """
        # Warm up camera and capture
        image_path = self.capture_image_from_webcam(
            output_path=output_path,
            warmup_frames=1,                # ⬅️ Increased warmup frames
            stabilization_time=0.1         # ⬅️ Slightly longer delay
        )

        if not image_path:
            return None

        # Run OCR
        text = self.extract_text(
            image_path,
            languages=languages,
            psm_mode=psm_mode,
            enhancement_level=enhancement_level
        )

        # Auto-delete the temporary image
        try:
            os.remove(image_path)
            logger.info(f"🗑️ Deleted temporary image: {image_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp image: {e}")

        return text





# Utility functions for module usage
def create_test_image(output_path: str = "test_image.png", 
                     size: Tuple[int, int] = (800, 400)) -> str:
    """
    Create a test image with various text elements for OCR testing.
    
    Args:
        output_path: Path for generated test image
        size: Image dimensions (width, height)
        
    Returns:
        Path to created test image
    """
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a standard font
    try:
        font = ImageFont.truetype("arial.ttf", 36)
        small_font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
        small_font = font
    
    # Add various text elements
    test_texts = [
        ("Tesseract OCR Test", 50, 50, font),
        ("Multi-language support", 50, 120, font),
        ("Numbers: 123456789", 50, 190, font),
        ("Email: test@example.com", 50, 260, small_font),
        ("Special chars: @#$%&*", 50, 320, small_font)
    ]
    
    for text, x, y, text_font in test_texts:
        draw.text((x, y), text, fill='black', font=text_font)
    
    img.save(output_path)
    logger.info(f"Test image created: {output_path}")
    return output_path


def verify_installation() -> bool:
    """
    Verify Tesseract OCR installation and required dependencies.
    
    Returns:
        True if installation is complete and functional
    """
    logger.info("Verifying Tesseract OCR installation")
    
    try:
        # Test Tesseract availability
        ocr = TesseractOCR()
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {version}")
        
        # Test language availability
        languages = ocr.available_languages
        logger.info(f"Available languages: {', '.join(languages)}")
        
        # Test dependencies
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
        
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
        
        logger.info("Installation verification successful")
        return True
        
    except Exception as e:
        logger.error(f"Installation verification failed: {e}")
        logger.info("Please ensure Tesseract OCR and required Python packages are installed")
        return False


# Example usage for testing
if __name__ == "__main__":
    ocr = TesseractOCR(camera_id=1)
    ocr_text = ocr.capture_and_extract_text()
    print("OCR Text:", ocr_text)