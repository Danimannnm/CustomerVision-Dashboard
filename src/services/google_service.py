"""
Google AutoML service implementation.
"""
import time
import logging
from typing import List

from google.cloud import automl
from google.auth.exceptions import DefaultCredentialsError

from ..models.detection_models import DetectionService, DetectionResult, Detection, BoundingBox
from ..utils.config import Config


class GoogleAutoMLService(DetectionService):
    """Google AutoML object detection service."""
    
    def __init__(self, config: Config):
        """Initialize Google AutoML service."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._prediction_client = None
        
    def _get_prediction_client(self):
        """Get or create prediction client."""
        if self._prediction_client is None:
            try:
                self._prediction_client = automl.PredictionServiceClient()
            except DefaultCredentialsError as e:
                self.logger.error(f"Google Cloud credentials not found: {e}")
                raise Exception("Google Cloud credentials not configured. Please set up authentication.")
        return self._prediction_client
    
    def detect_objects(self, image_bytes: bytes) -> DetectionResult:
        """Detect objects using Google AutoML API."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting Google AutoML object detection")
            
            prediction_client = self._get_prediction_client()
            
            # Create the endpoint path
            endpoint_path = f"projects/{self.config.google_project_id}/locations/{self.config.google_location}/endpoints/{self.config.google_endpoint_id}"
            
            # Prepare the image
            image = automl.Image(image_bytes=image_bytes)
            payload = automl.ExamplePayload(image=image)
            
            # Set parameters
            params = {"score_threshold": "0.3"}
            
            # Make prediction request
            request = automl.PredictRequest(
                name=endpoint_path,
                payload=payload,
                params=params
            )
            
            response = prediction_client.predict(request=request)
            
            processing_time = time.time() - start_time
            
            # Parse Google response
            detections = self._parse_google_response(response)
            
            # Get image dimensions (will be set by caller if needed)
            image_dimensions = (0, 0)
            
            detection_result = DetectionResult(
                service_name="Google AutoML",
                detections=detections,
                processing_time=processing_time,
                image_dimensions=image_dimensions,
                confidence_threshold=0.3
            )
            
            self.logger.info(f"Google detection completed in {processing_time:.2f}s with {len(detections)} detections")
            return detection_result
            
        except Exception as e:
            self.logger.error(f"Google AutoML detection failed: {e}")
            raise Exception(f"Google AutoML detection failed: {str(e)}")
    
    def _parse_google_response(self, response) -> List[Detection]:
        """Parse Google AutoML API response."""
        detections = []
        
        for result in response.payload:
            try:
                # Get bounding box coordinates
                bbox = result.image_object_detection.bounding_box
                normalized_vertices = bbox.normalized_vertices
                
                if len(normalized_vertices) >= 2:
                    # Calculate bounding box from normalized vertices
                    x_coords = [vertex.x for vertex in normalized_vertices]
                    y_coords = [vertex.y for vertex in normalized_vertices]
                    
                    left = min(x_coords)
                    top = min(y_coords)
                    width = max(x_coords) - left
                    height = max(y_coords) - top
                    
                    bounding_box = BoundingBox(
                        left=left,
                        top=top,
                        width=width,
                        height=height
                    )
                    
                    detection = Detection(
                        tag_name=result.display_name,
                        confidence=result.image_object_detection.score,
                        bounding_box=bounding_box
                    )
                    
                    detections.append(detection)
                    
            except Exception as e:
                self.logger.warning(f"Failed to parse Google detection: {e}")
                continue
                
        return detections
    
    def get_service_name(self) -> str:
        """Get the service name."""
        return "Google AutoML"
    
    def is_configured(self) -> bool:
        """Check if Google service is properly configured."""
        try:
            return self.config.validate_google_config()
        except Exception:
            return False
