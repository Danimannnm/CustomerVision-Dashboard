"""
Service factory for creating detection service instances.
"""
import logging
from typing import Optional, Dict, Any

from .azure_service import AzureCustomVisionService
from .google_service import GoogleAutoMLService
from ..models.detection_models import DetectionService
from ..utils.config import Config


class ServiceFactory:
    """Factory for creating detection service instances."""
    
    def __init__(self, config: Config):
        """Initialize service factory."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._services = {}
        
    def get_service(self, service_name: str) -> Optional[DetectionService]:
        """Get a detection service by name."""
        if service_name not in self._services:
            self._services[service_name] = self._create_service(service_name)
        
        return self._services[service_name]
    
    def _create_service(self, service_name: str) -> Optional[DetectionService]:
        """Create a new service instance."""
        try:
            if service_name == "Azure Custom Vision":
                if self.config.validate_azure_config():
                    return AzureCustomVisionService(self.config)
                else:
                    self.logger.warning("Azure Custom Vision not configured properly")
                    return None
                    
            elif service_name == "Google AutoML":
                if self.config.validate_google_config():
                    return GoogleAutoMLService(self.config)
                else:
                    self.logger.warning("Google AutoML not configured properly")
                    return None
                    
            else:
                self.logger.error(f"Unknown service: {service_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create service {service_name}: {e}")
            return None
    
    def get_available_services(self) -> list:
        """Get list of available services."""
        return self.config.get_available_services()
    
    def is_service_available(self, service_name: str) -> bool:
        """Check if a service is available."""
        return service_name in self.get_available_services()
