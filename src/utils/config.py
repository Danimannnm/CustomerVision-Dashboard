"""
Configuration management for the object detection dashboard.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass


class Config:
    """Configuration manager for environment variables."""
    
    def __init__(self):
        """Initialize configuration by loading environment variables."""
        load_dotenv()
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.DEBUG,  # Changed to DEBUG for better debugging
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('object_detection_dashboard.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    # Azure Custom Vision Configuration
    @property
    def azure_prediction_url(self) -> str:
        """Get Azure Custom Vision prediction URL."""
        url = os.getenv('CUSTOMVISION_PREDICTION_URL')
        if not url:
            raise ConfigurationError("CUSTOMVISION_PREDICTION_URL environment variable not set")
        return url
    
    @property
    def azure_prediction_key(self) -> str:
        """Get Azure Custom Vision prediction key."""
        key = os.getenv('CUSTOMVISION_PREDICTION_KEY')
        if not key:
            raise ConfigurationError("CUSTOMVISION_PREDICTION_KEY environment variable not set")
        return key
    
    # Google AutoML Configuration
    @property
    def google_project_id(self) -> str:
        """Get Google Cloud project ID."""
        project_id = os.getenv('GOOGLE_PROJECT_ID')
        if not project_id:
            raise ConfigurationError("GOOGLE_PROJECT_ID environment variable not set")
        return project_id
    
    @property
    def google_endpoint_id(self) -> str:
        """Get Google AutoML endpoint ID."""
        endpoint_id = os.getenv('GOOGLE_ENDPOINT_ID')
        if not endpoint_id:
            raise ConfigurationError("GOOGLE_ENDPOINT_ID environment variable not set")
        return endpoint_id
    
    @property
    def google_location(self) -> str:
        """Get Google Cloud location."""
        location = os.getenv('GOOGLE_LOCATION', 'us-central1')
        return location
    
    def validate_azure_config(self) -> bool:
        """Validate Azure configuration."""
        try:
            _ = self.azure_prediction_url
            _ = self.azure_prediction_key
            return True
        except ConfigurationError as e:
            self.logger.warning(f"Azure configuration invalid: {e}")
            return False
    
    def validate_google_config(self) -> bool:
        """Validate Google configuration."""
        try:
            _ = self.google_project_id
            _ = self.google_endpoint_id
            _ = self.google_location
            return True
        except ConfigurationError as e:
            self.logger.warning(f"Google configuration invalid: {e}")
            return False
    
    def get_available_services(self) -> list:
        """Get list of available services based on configuration."""
        services = []
        if self.validate_azure_config():
            services.append('Azure Custom Vision')
        if self.validate_google_config():
            services.append('Google AutoML')
        return services
