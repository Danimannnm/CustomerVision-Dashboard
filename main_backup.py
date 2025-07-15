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
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def _render_header(self):
        """Render the dashboard header."""
        st.title("🔍 Object Detection Dashboard")
        st.markdown("""
        Upload an image to detect objects using Azure Custom Vision and Google AutoML services.
        The dashboard provides detailed analytics and visualizations of the detection results.
        """)
        
        # Show available services
        available_services = self.service_factory.get_available_services()
        if available_services:
            st.success(f"✅ Available Services: {', '.join(available_services)}")
        else:
            st.error("❌ No detection services are configured. Please check your environment variables.")
            st.stop()
    
    def _render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.header("🎛️ Configuration")
        
        # Service selection with improved styling
        available_services = self.service_factory.get_available_services()
        st.sidebar.markdown("**🔧 Detection Service**")
        selected_service = st.sidebar.selectbox(
            "Choose Object Detection Service",
            options=available_services,
            help="Select which AI service to use for object detection"
        )
        selected_services = [selected_service] if selected_service else []
        
        st.sidebar.divider()
        
        # Service-specific configuration
        if selected_service == "Azure Custom Vision":
            st.sidebar.subheader("Azure Custom Vision Settings")
            
            # Get current config values
            prediction_url = st.sidebar.text_input(
                "Prediction URL",
                value=self.config.azure_prediction_url,
                help="Your Azure Custom Vision prediction endpoint URL"
            )
            
            prediction_key = st.sidebar.text_input(
                "Prediction Key",
                value=self.config.azure_prediction_key,
                type="password",
                help="Your Azure Custom Vision prediction key"
            )
            
            confidence_threshold = st.sidebar.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum confidence score for detections"
            )
            
            # Store in session state
            st.session_state.azure_config = {
                "prediction_url": prediction_url,
                "prediction_key": prediction_key,
                "confidence_threshold": confidence_threshold
            }
            
        elif selected_service == "Google AutoML":
            st.sidebar.subheader("Google AutoML Settings")
            
            project_id = st.sidebar.text_input(
                "Project ID",
                value=self.config.google_project_id,
                help="Your Google Cloud Project ID"
            )
            
            endpoint_id = st.sidebar.text_input(
                "Endpoint ID",
                value=self.config.google_endpoint_id,
                help="Your Vertex AI endpoint ID"
            )
            
            location = st.sidebar.text_input(
                "Location",
                value=self.config.google_location,
                help="Vertex AI location (e.g., europe-west4)"
            )
            
            confidence_threshold = st.sidebar.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum confidence score for detections"
            )
            
            st.sidebar.info("💡 Make sure you've configured Google Cloud authentication")
            
            # Store in session state
            st.session_state.google_config = {
                "project_id": project_id,
                "endpoint_id": endpoint_id,
                "location": location,
                "confidence_threshold": confidence_threshold
            }
        
        # Show threshold percentage
        if selected_service:
            st.sidebar.caption(f"Currently showing detections with ≥{confidence_threshold*100:.0f}% confidence")
        
        st.sidebar.divider()
        
        # Info section
        with st.sidebar.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **Object Detection Dashboard**
            
            This tool uses AI services to detect and identify objects in your images.
            
            - Upload any image (PNG, JPG, etc.)
            - Choose your preferred AI service
            - Adjust confidence threshold
            - View detailed results with bounding boxes
            """)
        
        # Store settings in session state
        st.session_state.selected_services = selected_services
        st.session_state.confidence_threshold = confidence_threshold
    
    def _render_main_content(self):
        """Render the main content area."""
        # Create two-column layout like the reference
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._render_file_upload()
            
            if st.session_state.uploaded_image is not None:
                # Detection section
                self._render_detection_section()
        
        with col2:
            self._render_results_section()
    
    def _render_file_upload(self):
        """Render the file upload section."""
        st.header("📁 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            help="Upload an image file for object detection"
        )
        
        if uploaded_file is not None:
            try:
                # Validate image
                image_bytes = uploaded_file.read()
                if not self.image_processor.validate_image(image_bytes):
                    st.error("❌ Invalid image format. Please upload a valid image file.")
                    return
                
                # Load and process image
                image = Image.open(io.BytesIO(image_bytes))
                image = self.image_processor.convert_to_rgb(image)
                
                # Store in session state
                st.session_state.uploaded_image = image
                st.session_state.uploaded_image_bytes = image_bytes
                
                # Display uploaded image
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Show image info
                image_info = self.image_processor.get_image_info(image)
                st.success(f"✅ Image loaded: {image_info['width']}x{image_info['height']} pixels")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")
                self.logger.error(f"Image processing error: {e}")
    
    def _render_detection_section(self):
        """Render the object detection section."""
        st.header("� Object Detection")
        
        # Check if service is selected
        if not st.session_state.selected_services:
            st.warning("⚠️ Please select a detection service from the sidebar.")
            return
        
        # Detection button with better styling
        if st.button("🔍 Detect Objects", type="primary", use_container_width=True):
            self._run_detection()
    
    def _run_detection(self):
        """Run object detection on the uploaded image."""
        if not st.session_state.selected_services:
            st.warning("⚠️ Please select at least one detection service.")
            return
        
        results = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_services = len(st.session_state.selected_services)
        
        for i, service_name in enumerate(st.session_state.selected_services):
            try:
                status_text.text(f"Running {service_name}...")
                
                # Get service
                service = self.service_factory.get_service(service_name)
                if service is None:
                    st.error(f"❌ Failed to initialize {service_name}")
                    continue
                
                # Run detection
                result = service.detect_objects(st.session_state.uploaded_image_bytes)
                
                # Update image dimensions
                result.image_dimensions = st.session_state.uploaded_image.size
                
                results.append(result)
                
                # Update progress
                progress_bar.progress((i + 1) / total_services)
                
                self.logger.info(f"Detection completed for {service_name}")
                
            except Exception as e:
                st.error(f"❌ {service_name} detection failed: {e}")
                self.logger.error(f"Detection error for {service_name}: {e}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if results:
            st.session_state.detection_results = results
            st.success(f"✅ Detection completed for {len(results)} service(s)")
            
            # Process images with bounding boxes
            self._process_detection_images(results)
        else:
            st.error("❌ No detection results obtained")
    
    def _process_detection_images(self, results: List[DetectionResult]):
        """Process images with bounding boxes."""
        try:
            # Create processed images for each service
            processed_images = {}
            
            for result in results:
                processed_image = self.image_processor.draw_bounding_boxes(
                    st.session_state.uploaded_image,
                    result,
                    st.session_state.confidence_threshold
                )
                processed_images[result.service_name] = processed_image
            
            st.session_state.processed_images = processed_images
            
        except Exception as e:
            st.error(f"❌ Error processing detection images: {e}")
            self.logger.error(f"Image processing error: {e}")
    
    def _render_results_section(self):
        """Render the detection results section."""
        st.header("📊 Detection Results")
        
        if hasattr(st.session_state, 'detection_results') and st.session_state.detection_results:
            # Get the current result (since we're using single service selection now)
            result = st.session_state.detection_results[0]
            service_name = result.service_name
            
            self._display_results(result, service_name)
            
            # Display annotated image if available
            if hasattr(st.session_state, 'processed_images') and st.session_state.processed_images:
                st.subheader("🎯 Annotated Image")
                processed_image = st.session_state.processed_images[service_name]
                st.image(
                    processed_image, 
                    caption="Image with detected objects and bounding boxes", 
                    use_container_width=True
                )
        else:
            st.info("Upload an image and click 'Detect Objects' to see results here.")
        
        # Create three columns: original image, processed image, and detection details
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state.get('show_original', True):
                st.subheader("� Original")
                display_image = self.image_processor.resize_image(
                    st.session_state.uploaded_image, 
                    max_width=300, 
                    max_height=250
                )
                st.image(display_image, caption="Original Image", use_container_width=False)
        
        with col2:
            st.subheader("🎯 Detected Objects")
            if 'processed_images' in st.session_state:
                processed_image = st.session_state.processed_images[result.service_name]
                display_image = self.image_processor.resize_image(
                    processed_image, 
                    max_width=300, 
                    max_height=250
                )
                st.image(display_image, caption=f"Detections from {result.service_name}", use_container_width=False)
                
                # Show color key
                self._show_color_key(result)
        
        with col3:
            # Detection details in expandable cards
            filtered_detections = result.get_high_confidence_detections(st.session_state.confidence_threshold)
            
            st.subheader("🔍 Detection Details")
            st.metric("Total Detections", len(filtered_detections))
            st.metric("Processing Time", f"{result.processing_time:.2f}s")
            
            # Group detections by tag
            tag_groups = {}
            for detection in filtered_detections:
                if detection.tag_name not in tag_groups:
                    tag_groups[detection.tag_name] = []
                tag_groups[detection.tag_name].append(detection)
            
            # Show expandable cards for each tag
            for tag_name, detections in tag_groups.items():
                with st.expander(f"📋 {tag_name} ({len(detections)} found)"):
                    for i, detection in enumerate(detections, 1):
                        st.markdown(f"""
                        **Instance {i}:**
                        - Confidence: {detection.confidence:.3f} ({detection.confidence*100:.1f}%)
                        - Position: ({detection.bounding_box.left:.3f}, {detection.bounding_box.top:.3f})
                        - Size: {detection.bounding_box.width:.3f} × {detection.bounding_box.height:.3f}
                        """)
    
    def _show_color_key(self, result: DetectionResult):
        """Show color key for bounding boxes."""
        filtered_detections = result.get_high_confidence_detections(st.session_state.confidence_threshold)
        unique_tags = list(set(d.tag_name for d in filtered_detections))
        
        if unique_tags:
            st.markdown("**Color Key:**")
            colors = self.image_processor.colors
            for i, tag in enumerate(unique_tags):
                color = colors[i % len(colors)]
                st.markdown(f"<span style='color: {color}; font-weight: bold;'>●</span> {tag}", unsafe_allow_html=True)

    def _show_color_key_improved(self, result: DetectionResult):
        """Show an improved color key for bounding boxes."""
        filtered_detections = result.get_high_confidence_detections(st.session_state.confidence_threshold)
        unique_tags = list(set(d.tag_name for d in filtered_detections))
        
        if unique_tags:
            colors = self.image_processor.colors
            cols = st.columns(min(3, len(unique_tags)))  # Max 3 columns
            
            for i, tag in enumerate(unique_tags):
                color = colors[i % len(colors)]
                col_idx = i % len(cols)
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style='
                        display: flex; 
                        align-items: center; 
                        padding: 5px; 
                        margin: 2px 0;
                        background-color: rgba(255,255,255,0.05);
                        border-radius: 3px;
                    '>
                        <div style='
                            width: 16px; 
                            height: 16px; 
                            background-color: {color}; 
                            border-radius: 2px; 
                            margin-right: 8px;
                        '></div>
                        <span style='font-size: 14px;'>{tag}</span>
                    </div>
                    """, unsafe_allow_html=True)


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
