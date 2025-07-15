"""
Image processing utilities for the object detection dashboard.
"""
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Tuple
import io

from ..models.detection_models import Detection, DetectionResult


class ImageProcessor:
    """Utility class for image processing operations."""
    
    def __init__(self):
        """Initialize image processor."""
        self.logger = logging.getLogger(__name__)
        self.colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
            "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"
        ]
    
    def draw_bounding_boxes(
        self, 
        image: Image.Image, 
        detection_result: DetectionResult,
        confidence_threshold: float = 0.3
    ) -> Image.Image:
        """Draw bounding boxes on the image."""
        try:
            # Create a copy of the image
            image_copy = image.copy()
            draw = ImageDraw.Draw(image_copy)
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Filter detections by confidence
            filtered_detections = detection_result.get_high_confidence_detections(confidence_threshold)
            
            # Get unique tags for color mapping
            unique_tags = list(set(d.tag_name for d in filtered_detections))
            tag_colors = {}
            for i, tag in enumerate(unique_tags):
                tag_colors[tag] = self.colors[i % len(self.colors)]
            
            # Try to load a clean, readable font
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("calibri.ttf", 12)
                except (OSError, IOError):
                    font = ImageFont.load_default()
            
            for detection in filtered_detections:
                bbox = detection.bounding_box
                color = tag_colors[detection.tag_name]
                
                # Convert normalized coordinates to pixel coordinates
                left = int(bbox.left * img_width)
                top = int(bbox.top * img_height)
                right = int((bbox.left + bbox.width) * img_width)
                bottom = int((bbox.top + bbox.height) * img_height)
                
                # Draw bounding box with thin line
                draw.rectangle([left, top, right, bottom], outline=color, width=2)
                
                # Prepare clean label text (without confidence score)
                label = f"{detection.tag_name}"
                
                # Get text dimensions
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Calculate text position (top-left of bounding box)
                text_x = left
                text_y = max(0, top - text_height - 2)  # Ensure text doesn't go above image
                
                # Draw semi-transparent background for text
                background_coords = [text_x - 1, text_y - 1, text_x + text_width + 2, text_y + text_height + 1]
                draw.rectangle(background_coords, fill=color + "CC")  # Add transparency
                
                # Draw text with high contrast
                draw.text((text_x, text_y), label, fill="white", font=font)
            
            self.logger.info(f"Drew {len(filtered_detections)} bounding boxes on image")
            return image_copy
            
        except Exception as e:
            self.logger.error(f"Failed to draw bounding boxes: {e}")
            return image
    
    def resize_image(self, image: Image.Image, max_width: int = 800, max_height: int = 600) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        try:
            # Get current dimensions
            width, height = image.size
            
            # Calculate aspect ratio
            aspect_ratio = width / height
            
            # Calculate new dimensions
            if width > max_width or height > max_height:
                if aspect_ratio > 1:  # Width is greater
                    new_width = max_width
                    new_height = int(max_width / aspect_ratio)
                else:  # Height is greater
                    new_height = max_height
                    new_width = int(max_height * aspect_ratio)
                
                # Resize the image
                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
                return resized_image
            
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to resize image: {e}")
            return image
    
    def validate_image(self, image_bytes: bytes) -> bool:
        """Validate if the uploaded file is a valid image."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()
            return True
        except Exception as e:
            self.logger.warning(f"Invalid image format: {e}")
            return False
    
    def get_image_info(self, image: Image.Image) -> dict:
        """Get basic information about the image."""
        try:
            return {
                'width': image.size[0],
                'height': image.size[1],
                'mode': image.mode,
                'format': image.format if hasattr(image, 'format') else 'Unknown'
            }
        except Exception as e:
            self.logger.error(f"Failed to get image info: {e}")
            return {}
    
    def convert_to_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB format if needed."""
        try:
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        except Exception as e:
            self.logger.error(f"Failed to convert image to RGB: {e}")
            return image
