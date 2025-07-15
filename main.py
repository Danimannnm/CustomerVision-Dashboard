"""
Main Streamlit application for Object Detection Dashboard.
"""
import streamlit as st
import logging
from PIL import Image
import io
from typing import List

from src.utils.config import Config
from src.services.service_factory import ServiceFactory
from src.utils.image_processor import ImageProcessor
from src.models.detection_models import DetectionResult


class ObjectDetectionDashboard:
    """Main dashboard application class."""
    
    def __init__(self):
        """Initialize the dashboard."""
        try:
            self.config = Config()
            self.service_factory = ServiceFactory(self.config)
            self.image_processor = ImageProcessor()
            self.logger = logging.getLogger(__name__)
            
            # Initialize session state
            self._initialize_session_state()
            
        except Exception as e:
            st.error(f"Failed to initialize dashboard: {e}")
            st.stop()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = []
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = None
    
    def run(self):
        """Run the main dashboard application."""
        self._setup_page_config()
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Object Detection Dashboard",
            page_icon="🔍",
            layout="wide"
        )
    
    def _render_header(self):
        """Render the dashboard header."""
        st.title("🔍 Object Detection Dashboard")
        st.markdown("Choose a service and upload an image to detect objects")
        
        # Show available services
        available_services = self.service_factory.get_available_services()
        if not available_services:
            st.error("❌ No detection services are configured. Please check your environment variables.")
            st.stop()
    
    def _render_sidebar(self):
        """Render the sidebar with controls."""
        with st.sidebar:
            st.header("Configuration")
            
            # Service selection
            available_services = self.service_factory.get_available_services()
            service_choice = st.selectbox(
                "Choose Object Detection Service",
                available_services,
                help="Select which AI service to use for object detection"
            )
            
            st.divider()
            
            # Service-specific configuration
            if service_choice == "Azure Custom Vision":
                st.subheader("Azure Custom Vision Settings")
                
                # Get current config values
                prediction_url = st.text_input(
                    "Prediction URL",
                    value=self.config.azure_prediction_url,
                    help="Your Azure Custom Vision prediction endpoint URL"
                )
                
                prediction_key = st.text_input(
                    "Prediction Key",
                    value=self.config.azure_prediction_key,
                    type="password",
                    help="Your Azure Custom Vision prediction key"
                )
                
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Minimum confidence score for detections (50% default)"
                )
                
                # Store in session state
                st.session_state.azure_config = {
                    "prediction_url": prediction_url,
                    "prediction_key": prediction_key,
                    "confidence_threshold": confidence_threshold
                }
                
            elif service_choice == "Google AutoML":
                st.subheader("Google AutoML Settings")
                
                project_id = st.text_input(
                    "Project ID",
                    value=self.config.google_project_id,
                    help="Your Google Cloud Project ID"
                )
                
                endpoint_id = st.text_input(
                    "Endpoint ID",
                    value=self.config.google_endpoint_id,
                    help="Your Vertex AI endpoint ID"
                )
                
                location = st.text_input(
                    "Location",
                    value=self.config.google_location,
                    help="Vertex AI location (e.g., europe-west4)"
                )
                
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Minimum confidence score for detections"
                )
                
                st.info("💡 Make sure you've run `gcloud auth application-default login` for authentication")
                
                # Store in session state
                st.session_state.google_config = {
                    "project_id": project_id,
                    "endpoint_id": endpoint_id,
                    "location": location,
                    "confidence_threshold": confidence_threshold
                }
            
            # Store service choice
            st.session_state.selected_service = service_choice
            st.session_state.confidence_threshold = confidence_threshold
    
    def _render_main_content(self):
        """Render the main content area."""
        # Create two-column layout like the reference
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._render_file_upload()
        
        with col2:
            self._render_results_section()
    
    def _render_file_upload(self):
        """Render the file upload section."""
        st.header("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image file for object detection"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Store in session state
            st.session_state.uploaded_image = image
            st.session_state.uploaded_file = uploaded_file
            
            # Detection button
            if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
                self._run_detection(uploaded_file)
    
    def _run_detection(self, uploaded_file):
        """Run object detection on the uploaded image."""
        service_choice = st.session_state.selected_service
        
        with st.spinner(f"Processing with {service_choice}..."):
            try:
                results, annotated_image = self._process_image(uploaded_file, service_choice)
                st.session_state.detection_results = results
                st.session_state.annotated_image = annotated_image
                st.success("Detection completed!")
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
    
    def _process_image(self, uploaded_file, service_choice):
        """Process the uploaded image with the selected service."""
        # Get service
        service = self.service_factory.get_service(service_choice)
        if service is None:
            raise ValueError(f"Failed to initialize {service_choice}")
        
        # Convert uploaded file to bytes
        image_bytes = uploaded_file.getvalue()
        
        # Run detection with confidence threshold
        result = service.detect_objects(image_bytes, st.session_state.confidence_threshold)
        
        # Update image dimensions
        result.image_dimensions = st.session_state.uploaded_image.size
        
        # Create annotated image
        annotated_image = self.image_processor.draw_bounding_boxes(
            st.session_state.uploaded_image,
            result,
            st.session_state.confidence_threshold
        )
        
        return result, annotated_image
    
    def _render_results_section(self):
        """Render the detection results section."""
        st.header("Detection Results")
        
        if hasattr(st.session_state, 'detection_results') and st.session_state.detection_results:
            result = st.session_state.detection_results
            service_name = st.session_state.selected_service
            
            self._display_results(result, service_name)
            
            # Display annotated image if available
            if hasattr(st.session_state, 'annotated_image') and st.session_state.annotated_image:
                st.subheader("🎯 Annotated Image")
                st.image(
                    st.session_state.annotated_image, 
                    caption="Image with detected objects and bounding boxes", 
                    use_container_width=True
                )
        else:
            st.info("Upload an image and click 'Detect Objects' to see results here.")
    
    def _display_results(self, result: DetectionResult, service_name: str):
        """Display detection results in a formatted way like the reference."""
        # Filter detections by confidence threshold
        filtered_detections = result.get_high_confidence_detections(st.session_state.confidence_threshold)
        
        if not filtered_detections:
            st.warning("No objects detected in the image with the current confidence threshold.")
            return
        
        st.subheader(f"Results from {service_name}")
        
        # Show color coding info if multiple tags
        unique_tags = list(set(d.tag_name for d in filtered_detections))
        if len(unique_tags) > 1:
            st.info(f"🎨 Each tag has a unique color: {', '.join(unique_tags)}")
        
        # Display summary metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Objects Detected", len(filtered_detections))
        with col2:
            threshold = st.session_state.confidence_threshold
            st.metric("Confidence Threshold", f"{threshold:.0%}")
        
        # Display detailed results for each detection
        for i, detection in enumerate(filtered_detections, 1):
            with st.expander(f"Detection {i}: {detection.tag_name}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    confidence = detection.confidence
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.write(f"**Class:** {detection.tag_name}")
                    
                    # Color code confidence
                    if confidence >= 0.8:
                        st.success("High confidence")
                    elif confidence >= 0.6:
                        st.warning("Medium confidence")
                    else:
                        st.info("Low confidence")
                
                with col2:
                    bbox = detection.bounding_box
                    st.write("**Bounding Box (Normalized):**")
                    st.write(f"Left: {bbox.left:.3f}")
                    st.write(f"Top: {bbox.top:.3f}")
                    st.write(f"Width: {bbox.width:.3f}")
                    st.write(f"Height: {bbox.height:.3f}")
        
        # Display raw JSON for debugging
        with st.expander("Raw Results (JSON)", expanded=False):
            # Convert detection results to a JSON-serializable format
            json_results = []
            for detection in filtered_detections:
                json_results.append({
                    "tag_name": detection.tag_name,
                    "confidence": detection.confidence,
                    "bounding_box": {
                        "left": detection.bounding_box.left,
                        "top": detection.bounding_box.top,
                        "width": detection.bounding_box.width,
                        "height": detection.bounding_box.height
                    }
                })
            st.json(json_results)


def main():
    """Main application entry point."""
    try:
        dashboard = ObjectDetectionDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
