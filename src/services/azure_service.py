"""
Azure Custom Vision service implementation.
"""
import time
import logging
import requests
from typing import List
from io import BytesIO

from ..models.detection_models import DetectionService, DetectionResult, Detection, BoundingBox
from ..utils.config import Config


class AzureCustomVisionService(DetectionService):
    """Azure Custom Vision object detection service."""
    
    def __init__(self, config: Config):
        """Initialize Azure Custom Vision service."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def detect_objects(self, image_bytes: bytes, confidence_threshold: float = 0.3) -> DetectionResult:
        """Detect objects using Azure Custom Vision API."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting Azure Custom Vision object detection")
            
            headers = {
                'Prediction-Key': self.config.azure_prediction_key,
                'Content-Type': 'application/octet-stream'
            }
            
            response = requests.post(
                self.config.azure_prediction_url,
                headers=headers,
                data=image_bytes,
                timeout=30
            )
            
            response.raise_for_status()
            result_data = response.json()
            
            self.logger.info(f"Azure response: {result_data}")
            
            processing_time = time.time() - start_time
            
            # Parse Azure response
            detections = self._parse_azure_response(result_data)
            self.logger.info(f"Azure parsed {len(detections)} detections")
            
            # Get image dimensions (will be set by caller if needed)
            image_dimensions = (0, 0)
            
            detection_result = DetectionResult(
                service_name="Azure Custom Vision",
                detections=detections,
                processing_time=processing_time,
                image_dimensions=image_dimensions,
                confidence_threshold=confidence_threshold
            )
            
            self.logger.info(f"Azure detection completed in {processing_time:.2f}s with {len(detections)} detections")
            return detection_result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Azure API request failed: {e}")
            raise Exception(f"Azure Custom Vision API error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Azure detection failed: {e}")
            raise Exception(f"Azure Custom Vision detection failed: {str(e)}")
    
    def _parse_azure_response(self, response_data: dict) -> List[Detection]:
        """Parse Azure Custom Vision API response."""
        detections = []
        
        self.logger.info(f"Parsing Azure response: {response_data}")
        
        predictions = response_data.get('predictions', [])
        self.logger.info(f"Found {len(predictions)} predictions in Azure response")
        
        for i, prediction in enumerate(predictions):
            try:
                self.logger.debug(f"Processing Azure prediction {i}: {prediction}")
                
                bbox_data = prediction.get('boundingBox', {})
                tag_name = prediction.get('tagName', 'Unknown')
                confidence = prediction.get('probability', 0.0)
                
                self.logger.debug(f"Azure detection: {tag_name} ({confidence:.3f}) bbox: {bbox_data}")
                
                # Validate bounding box data
                if not bbox_data:
                    self.logger.warning(f"No bounding box data for prediction {i}")
                    continue
                
                bounding_box = BoundingBox(
                    left=float(bbox_data.get('left', 0.0)),
                    top=float(bbox_data.get('top', 0.0)),
                    width=float(bbox_data.get('width', 0.0)),
                    height=float(bbox_data.get('height', 0.0))
                )
                
                # Validate box dimensions
                if bounding_box.width <= 0 or bounding_box.height <= 0:
                    self.logger.warning(f"Invalid box dimensions for prediction {i}: width={bounding_box.width}, height={bounding_box.height}")
                    continue
                
                detection = Detection(
                    tag_name=tag_name,
                    confidence=confidence,
                    bounding_box=bounding_box
                )
                
                detections.append(detection)
                self.logger.debug(f"Successfully added Azure detection: {tag_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to parse Azure detection {i}: {e}")
                continue
        
        self.logger.info(f"Successfully parsed {len(detections)} Azure detections")
        return detections
    
    def get_service_name(self) -> str:
        """Get the service name."""
        return "Azure Custom Vision"
    
    def is_configured(self) -> bool:
        """Check if Azure service is properly configured."""
        return self.config.validate_azure_config()
