#!/usr/bin/env python3
"""
License Plate Recognition Module
Handles image processing and license plate text recognition.
"""

import logging
import os
import itertools
from typing import Optional, List, Tuple
from functools import lru_cache

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks, rotate

from config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlateRecognizer:
    """Handles license plate detection and text recognition."""
    
    def __init__(self):
        """Initialize the recognizer with models."""
        self.config = get_config()
        self.detection_interpreter = None
        self.recognition_interpreter = None
        self._load_models()
    
    def _load_models(self):
        """Load TensorFlow Lite models."""
        try:
            # Load detection model
            if os.path.exists(self.config.models.resnet_path):
                self.detection_interpreter = tf.lite.Interpreter(
                    model_path=self.config.models.resnet_path
                )
                self.detection_interpreter.allocate_tensors()
                logger.info("Detection model loaded successfully")
            else:
                raise FileNotFoundError(f"Detection model not found: {self.config.models.resnet_path}")
            
            # Load recognition model
            if os.path.exists(self.config.models.recognition_path):
                self.recognition_interpreter = tf.lite.Interpreter(
                    model_path=self.config.models.recognition_path
                )
                self.recognition_interpreter.allocate_tensors()
                logger.info("Recognition model loaded successfully")
            else:
                raise FileNotFoundError(f"Recognition model not found: {self.config.models.recognition_path}")
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    @lru_cache(maxsize=1)
    def get_alphabet(self) -> List[str]:
        """Get the recognition alphabet (cached)."""
        return "0 1 2 3 4 5 6 7 8 9 A B C E H K M O P T X Y".split()
    
    def decode_batch(self, output: np.ndarray) -> List[str]:
        """Decode model output to text."""
        letters = self.get_alphabet()
        results = []
        
        for j in range(output.shape[0]):
            out_best = list(np.argmax(output[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = "".join([letters[c] for c in out_best if c < len(letters)])
            results.append(outstr)
        
        return results
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection."""
        if image is None:
            raise ValueError("Image is None")
        
        # Resize to model input size
        image = cv2.resize(image, (self.config.processing.image_size, self.config.processing.image_size))
        image = image.astype(np.float32)
        return image
    
    def detect_plate(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect license plate in image and return bounding box coordinates."""
        try:
            processed_image = self.preprocess_image(image)
            
            # Prepare input
            input_details = self.detection_interpreter.get_input_details()
            output_details = self.detection_interpreter.get_output_details()
            
            X_data = np.float32(processed_image.reshape(1, self.config.processing.image_size, 
                                                      self.config.processing.image_size, 3))
            
            # Run inference
            self.detection_interpreter.set_tensor(input_details[0]["index"], X_data)
            self.detection_interpreter.invoke()
            detection = self.detection_interpreter.get_tensor(output_details[0]["index"])
            
            # Extract bounding box
            if np.min(detection[0, 0, :]) >= 0:
                image_height, image_width = image.shape[:2]
                box_x = int(detection[0, 0, 0] * image_height)
                box_y = int(detection[0, 0, 1] * image_width)
                box_width = int(detection[0, 0, 2] * image_height)
                box_height = int(detection[0, 0, 3] * image_width)
                
                return (box_x, box_y, box_width, box_height)
            
            return None
            
        except Exception as e:
            logger.error(f"Plate detection failed: {e}")
            return None
    
    def correct_rotation(self, image_crop: np.ndarray) -> np.ndarray:
        """Correct rotation of cropped license plate image."""
        try:
            # Convert to grayscale and detect edges
            grayscale = rgb2gray(image_crop)
            edges = canny(grayscale, sigma=3.0)
            
            # Detect lines using Hough transform
            out, angles, distances = hough_line(edges)
            _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
            
            if len(angles_peaks) == 0:
                return image_crop
            
            angle = np.mean(np.rad2deg(angles_peaks))
            
            # Determine rotation angle
            if 0 <= angle <= 90:
                rot_angle = angle - 90
            elif -45 <= angle < 0:
                rot_angle = angle - 90
            elif -90 <= angle < -45:
                rot_angle = 90 + angle
            else:
                rot_angle = 0
            
            # Only rotate if angle is reasonable
            if abs(rot_angle) > self.config.processing.rotation_threshold:
                rot_angle = 0
            
            # Rotate image
            rotated = rotate(image_crop, rot_angle, resize=True) * 255
            rotated = rotated.astype(np.uint8)
            
            # Crop to remove rotation artifacts
            minus = np.abs(int(np.sin(np.radians(rot_angle)) * rotated.shape[0]))
            if (rotated.shape[1] / rotated.shape[0] < self.config.processing.crop_min_ratio 
                and minus > 10):
                rotated = rotated[minus:-minus, :, :]
            
            return rotated
            
        except Exception as e:
            logger.error(f"Rotation correction failed: {e}")
            return image_crop
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(
                clipLimit=self.config.processing.clahe_clip_limit,
                tileGridSize=self.config.processing.clahe_tile_grid_size
            )
            cl = clahe.apply(l)
            
            # Merge channels back
            enhanced = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            logger.error(f"Contrast enhancement failed: {e}")
            return image
    
    def recognize_text(self, image: np.ndarray) -> Optional[str]:
        """Recognize text from preprocessed license plate image."""
        try:
            # Prepare image for text recognition
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.config.processing.final_img_width, 
                                 self.config.processing.final_img_height))
            img = img.astype(np.float32) / 255.0
            img = img.T  # Transpose
            
            # Prepare input tensor
            input_details = self.recognition_interpreter.get_input_details()
            output_details = self.recognition_interpreter.get_output_details()
            
            X_data = np.float32(img.reshape(1, self.config.processing.final_img_width, 
                                          self.config.processing.final_img_height, 1))
            
            # Run inference
            self.recognition_interpreter.set_tensor(input_details[0]["index"], X_data)
            self.recognition_interpreter.invoke()
            net_out_value = self.recognition_interpreter.get_tensor(output_details[0]["index"])
            
            # Decode results
            pred_texts = self.decode_batch(net_out_value)
            return pred_texts[0] if pred_texts else None
            
        except Exception as e:
            logger.error(f"Text recognition failed: {e}")
            return None
    
    def process_image(self, image_path: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Complete image processing pipeline.
        
        Returns:
            Tuple of (recognized_text, processed_image_with_bbox)
        """
        try:
            # Read image
            image = cv2.imread(image_path, 1)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            logger.info(f"Processing image: {os.path.basename(image_path)}")
            
            # Detect license plate
            bbox = self.detect_plate(image)
            if bbox is None:
                logger.warning("No license plate detected in image")
                return None, image
            
            box_x, box_y, box_width, box_height = bbox
            
            # Draw bounding box on original image
            result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.rectangle(
                result_image,
                (box_y, box_x),
                (box_height, box_width),
                (230, 230, 21),
                thickness=3
            )
            
            # Crop license plate region
            image_crop = image[box_x:box_width, box_y:box_height, :]
            if image_crop.size == 0:
                logger.warning("Invalid crop region")
                return None, result_image
            
            # Correct rotation
            corrected_crop = self.correct_rotation(image_crop)
            
            # Enhance contrast
            enhanced_crop = self.enhance_contrast(corrected_crop)
            
            # Recognize text
            recognized_text = self.recognize_text(enhanced_crop)
            
            if recognized_text:
                logger.info(f"Recognized license plate: {recognized_text}")
            else:
                logger.warning("Could not recognize text from license plate")
            
            return recognized_text, result_image
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return None, None

# Global recognizer instance
_recognizer = None

def get_recognizer() -> PlateRecognizer:
    """Get the global recognizer instance."""
    global _recognizer
    if _recognizer is None:
        _recognizer = PlateRecognizer()
    return _recognizer
