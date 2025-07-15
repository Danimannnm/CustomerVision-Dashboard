import time
import logging
import base64
from typing import List

from google.auth.exceptions import DefaultCredentialsError
from google.cloud.aiplatform.gapic import PredictionServiceClient
from google.cloud.aiplatform.gapic.schema import predict

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
            detections = self._parse_response(response.predictions)
            detection_result = DetectionResult(
                service_name="Google AutoML",
                detections=detections,
                processing_time=processing_time,
                image_dimensions=(0, 0),
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
            return detections
        # predictions[0] contains keys: 'displayNames', 'bboxes', 'scores'
        result = predictions[0]
        names = result.get("displayNames", [])
        boxes = result.get("bboxes", [])
        scores = result.get("scores", [])
        for name, box, score in zip(names, boxes, scores):
            try:
                # box format: [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box
                left = xmin
                top = ymin
                width = xmax - xmin
                height = ymax - ymin
                bounding_box = BoundingBox(left=left, top=top, width=width, height=height)
                detections.append(Detection(tag_name=name, confidence=score, bounding_box=bounding_box))
            except Exception as e:
                self.logger.warning(f"Failed to parse prediction entry: {e}")
        return detections

    def get_service_name(self) -> str:
        return "Google AutoML"

    def is_configured(self) -> bool:
        try:
            return self.config.validate_google_config()
        except Exception:
            return False
