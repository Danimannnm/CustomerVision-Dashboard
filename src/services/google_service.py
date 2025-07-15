import time
import logging
import base64
from typing import List

from google.auth.exceptions import DefaultCredentialsError
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.cloud.aiplatform.gapic.schema import predict
from PIL import Image
import io

from ..models.detection_models import DetectionService, DetectionResult, Detection, BoundingBox
from ..utils.config import Config


class GoogleAutoMLService(DetectionService):
    """Google AutoML (Vertex AI) object detection service."""
    def __init__(self, config: Config):
        """Initialize Google AutoML service."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._prediction_client = None

    def _get_prediction_client(self) -> PredictionServiceClient:
        """Get or create Vertex AI PredictionServiceClient."""
        if self._prediction_client is None:
            try:
                client_options = {
                    "api_endpoint": f"{self.config.google_location}-aiplatform.googleapis.com"
                }
                self._prediction_client = PredictionServiceClient(client_options=client_options)
            except DefaultCredentialsError as e:
                self.logger.error(f"Google Cloud credentials not found: {e}")
                raise Exception("Google Cloud credentials not configured. Please set up authentication.")
        return self._prediction_client

    def detect_objects(self, image_bytes: bytes, confidence_threshold: float = 0.5) -> DetectionResult:
        """Detect objects using Vertex AI Prediction API."""
        start_time = time.time()
        self.logger.info("Starting Vertex AI object detection")
        try:
            # Get image dimensions for later use
            image = Image.open(io.BytesIO(image_bytes))
            img_width, img_height = image.size

            client = self._get_prediction_client()
            # Build the endpoint path
            endpoint = client.endpoint_path(
                project=self.config.google_project_id,
                location=self.config.google_location,
                endpoint=self.config.google_endpoint_id
            )
            # Prepare instance
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            instance = predict.instance.ImageObjectDetectionPredictionInstance(
                content=encoded
            ).to_value()
            instances = [instance]
            # Prepare parameters
            params = predict.params.ImageObjectDetectionPredictionParams(
                confidence_threshold=confidence_threshold
            ).to_value()
            # Call predict
            response = client.predict(
                endpoint=endpoint,
                instances=instances,
                parameters=params
            )
            processing_time = time.time() - start_time
            # Parse predictions
            self.logger.info(f"Raw response from Vertex AI: {response}")
            self.logger.info(f"Predictions: {response.predictions}")
            
            detections = self._parse_response(response.predictions)
            self.logger.info(f"Parsed {len(detections)} detections")
            
            detection_result = DetectionResult(
                service_name="Google AutoML",
                detections=detections,
                processing_time=processing_time,
                image_dimensions=(img_width, img_height),
                confidence_threshold=confidence_threshold
            )
            self.logger.info(f"Vertex AI detection completed in {processing_time:.2f}s with {len(detections)} detections")
            return detection_result
        except Exception as e:
            self.logger.error(f"Vertex AI detection failed: {e}")
            raise Exception(f"Vertex AI detection failed: {str(e)}")

    def _parse_response(self, predictions: List[dict]) -> List[Detection]:
        """Parse Vertex AI prediction response into Detection objects."""
        detections: List[Detection] = []
        
        if not predictions:
            self.logger.warning("No predictions received from Vertex AI")
            return detections
            
        self.logger.info(f"Processing {len(predictions)} predictions")
        
        for i, prediction in enumerate(predictions):
            self.logger.info(f"Prediction {i}: {prediction}")
            
            try:
                # Handle proto.marshal objects
                if hasattr(prediction, '_pb'):
                    # Extract the protobuf data
                    pb_data = prediction._pb
                    self.logger.info(f"Found protobuf data: {pb_data}")
                    # Use helper to extract list values from protobuf ListValue
                    # Extract confidences
                    conf_val = pb_data['confidences'] if 'confidences' in pb_data else None
                    confidences = self._extract_list_values(conf_val.list_value if conf_val and hasattr(conf_val, 'list_value') else None)
                    # Extract display names
                    name_val = pb_data['displayNames'] if 'displayNames' in pb_data else None
                    display_names = self._extract_list_values(name_val.list_value if name_val and hasattr(name_val, 'list_value') else None)
                    # Extract bounding boxes (each value is a ListValue)
                    bboxes_raw = []
                    bbox_val = pb_data['bboxes'] if 'bboxes' in pb_data else None
                    if bbox_val and hasattr(bbox_val, 'list_value'):
                        for item in bbox_val.list_value.values:
                            if hasattr(item, 'list_value'):
                                coords = self._extract_list_values(item.list_value)
                                bboxes_raw.append(coords)
                    self.logger.info(f"Extracted - confidences: {confidences}, names: {display_names}, bboxes: {bboxes_raw}")
                    # Validate we have data
                    if not confidences or not display_names or not bboxes_raw:
                        self.logger.warning(f"Missing data after extraction: conf={len(confidences)}, names={len(display_names)}, boxes={len(bboxes_raw)}")
                    # Create detections directly from extracted data
                    for j, (name, bbox, confidence) in enumerate(zip(display_names, bboxes_raw, confidences)):
                        try:
                            self.logger.debug(f"Creating detection {j}: {name} with confidence {confidence} and bbox {bbox}")
                            detection = self._create_detection_from_box(name, bbox, confidence)
                            if detection:
                                detections.append(detection)
                                self.logger.debug(f"Successfully added detection: {detection.tag_name}")
                            else:
                                self.logger.warning(f"Failed to create detection for {name}")
                        except Exception as e:
                            self.logger.warning(f"Failed to create detection: {e}")
                else:
                    # Fallback to standard dict parsing
                    # Convert prediction to dict if needed
                    if hasattr(prediction, 'to_dict'):
                        pred_dict = prediction.to_dict()
                    elif hasattr(prediction, '__dict__'):
                        pred_dict = prediction.__dict__
                    else:
                        pred_dict = dict(prediction)
                    
                    self.logger.info(f"Converted prediction dict: {pred_dict}")
                    
                    # Try different possible response formats
                    if 'displayNames' in pred_dict and 'bboxes' in pred_dict:
                        names = pred_dict.get("displayNames", [])
                        boxes = pred_dict.get("bboxes", [])
                        scores = pred_dict.get("confidences", pred_dict.get("scores", []))
                        
                        self.logger.info(f"Found standard format: {len(names)} names, {len(boxes)} boxes, {len(scores)} scores")
                        
                        for name, box, score in zip(names, boxes, scores):
                            try:
                                detection = self._create_detection_from_box(name, box, score)
                                if detection:
                                    detections.append(detection)
                            except Exception as e:
                                self.logger.warning(f"Failed to parse detection: {e}")
                    else:
                        self.logger.warning(f"Unknown prediction format, keys: {list(pred_dict.keys())}")
                    
            except Exception as e:
                self.logger.error(f"Failed to parse prediction {i}: {e}")
                continue
        
        self.logger.info(f"Successfully parsed {len(detections)} total detections")
        return detections
    
    def _extract_list_values(self, list_value_obj):
        """Extract values from protobuf ListValue object."""
        if not list_value_obj:
            return []
        
        try:
            values = []
            # Handle direct ListValue objects
            if hasattr(list_value_obj, 'values'):
                for value in list_value_obj.values:
                    if hasattr(value, 'number_value'):
                        values.append(value.number_value)
                    elif hasattr(value, 'string_value'):
                        values.append(value.string_value)
                    elif hasattr(value, 'list_value'):
                        # Nested list - extract recursively
                        nested_values = self._extract_list_values(value.list_value)
                        values.append(nested_values)
                    else:
                        # Try to get the value directly
                        values.append(value)
            # Handle when it's already the values attribute
            elif hasattr(list_value_obj, '__iter__'):
                for value in list_value_obj:
                    if hasattr(value, 'number_value'):
                        values.append(value.number_value)
                    elif hasattr(value, 'string_value'):
                        values.append(value.string_value)
                    elif hasattr(value, 'list_value'):
                        # Nested list - extract recursively
                        nested_values = self._extract_list_values(value.list_value)
                        values.append(nested_values)
                    else:
                        values.append(value)
            
            self.logger.debug(f"Extracted values: {values}")
            return values
        except Exception as e:
            self.logger.error(f"Failed to extract list values: {e}")
            return []
    
    def _create_detection_from_box(self, name: str, box: List[float], score: float) -> Detection:
        """Create a Detection object from name, box coordinates, and score."""
        try:
            if len(box) >= 4:
                # Google's box format: [xmin, xmax, ymin, ymax] (normalized)
                xmin, xmax, ymin, ymax = box[:4]
                
                # Ensure coordinates are between 0 and 1
                xmin = max(0.0, min(1.0, float(xmin)))
                ymin = max(0.0, min(1.0, float(ymin)))
                xmax = max(0.0, min(1.0, float(xmax)))
                ymax = max(0.0, min(1.0, float(ymax)))
                
                # Calculate width and height
                width = xmax - xmin
                height = ymax - ymin
                
                # Validate dimensions
                if width > 0 and height > 0:
                    bounding_box = BoundingBox(
                        left=xmin,
                        top=ymin,
                        width=width,
                        height=height
                    )
                    
                    detection = Detection(
                        tag_name=str(name),
                        confidence=float(score),
                        bounding_box=bounding_box
                    )
                    
                    self.logger.debug(f"Created detection: {name} ({score:.3f}) at [{xmin:.3f}, {ymin:.3f}, {width:.3f}, {height:.3f}]")
                    return detection
                else:
                    self.logger.warning(f"Invalid box dimensions: width={width}, height={height}")
            else:
                self.logger.warning(f"Box has insufficient coordinates: {box}")
                
        except Exception as e:
            self.logger.error(f"Failed to create detection from box: {e}")
            
        return None

    def get_service_name(self) -> str:
        return "Google AutoML"

    def is_configured(self) -> bool:
        try:
            return self.config.validate_google_config()
        except Exception:
            return False
