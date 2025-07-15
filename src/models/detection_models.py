"""
Data models for object detection results.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class BoundingBox:
    """Represents a bounding box for object detection."""
    left: float
    top: float
    width: float
    height: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height
        }


@dataclass
class Detection:
    """Represents a single object detection result."""
    tag_name: str
    confidence: float
    bounding_box: BoundingBox
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'tag_name': self.tag_name,
            'confidence': self.confidence,
            'bounding_box': self.bounding_box.to_dict()
        }


@dataclass
class DetectionResult:
    """Represents the complete detection result from a service."""
    service_name: str
    detections: List[Detection]
    processing_time: float
    image_dimensions: tuple
    confidence_threshold: float = 0.0
    
    def get_high_confidence_detections(self, threshold: float = None) -> List[Detection]:
        """Filter detections by confidence threshold."""
        thresh = threshold if threshold is not None else self.confidence_threshold
        return [d for d in self.detections if d.confidence >= thresh]
    
    def get_detection_count(self) -> int:
        """Get total number of detections."""
        return len(self.detections)
    
    def get_unique_tags(self) -> List[str]:
        """Get list of unique detected object tags."""
        return list(set(d.tag_name for d in self.detections))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'service_name': self.service_name,
            'detections': [d.to_dict() for d in self.detections],
            'processing_time': self.processing_time,
            'image_dimensions': self.image_dimensions,
            'confidence_threshold': self.confidence_threshold
        }


class DetectionService(ABC):
    """Abstract base class for object detection services."""
    
    @abstractmethod
    def detect_objects(self, image_bytes: bytes) -> DetectionResult:
        """Detect objects in the given image."""
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get the name of the detection service."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the service is properly configured."""
        pass
