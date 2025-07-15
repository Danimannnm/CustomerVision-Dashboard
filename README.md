# Object Detection Dashboard

A comprehensive Streamlit-based dashboard for object detection using Azure Custom Vision and Google AutoML services. The application provides an intuitive interface for uploading images, running object detection, and analyzing results with detailed visualizations.

## Features

- 🔍 **Multi-Service Support**: Integrates Azure Custom Vision and Google AutoML
- 📸 **Image Upload**: Support for multiple image formats (PNG, JPG, JPEG, BMP, WebP)
- 🎯 **Visual Detection**: Displays bounding boxes with confidence scores
- 📊 **Analytics Dashboard**: Comprehensive charts and statistics
- ⚙️ **Configurable**: Adjustable confidence thresholds and service selection
- 💾 **Export Functionality**: Download results as CSV
- 🔧 **Robust Logging**: Detailed error handling and troubleshooting

## Architecture

The application follows SOLID principles with a modular architecture:

```
src/
├── models/          # Data models and abstract interfaces
├── services/        # Service implementations for Azure and Google
├── utils/           # Utility classes for image processing and analytics
└── ui/              # User interface components
```

## Prerequisites

- Python 3.8+
- Azure Custom Vision subscription (optional)
- Google Cloud Project with AutoML enabled (optional)

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd ConsildatedFrontendv2
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate  # Windows
   # source venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Environment Variables

Create or update the `.env` file with your service credentials:

```properties
# Azure Custom Vision Configuration
CUSTOMVISION_PREDICTION_URL=https://your-region.api.cognitive.microsoft.com/customvision/v3.0/Prediction/your-project-id/detect/iterations/your-iteration/image
CUSTOMVISION_PREDICTION_KEY=your-prediction-key

# Google AutoML Configuration
GOOGLE_PROJECT_ID=your-project-id
GOOGLE_ENDPOINT_ID=your-endpoint-id
GOOGLE_LOCATION=your-region
```

### Azure Custom Vision Setup

1. Create a Custom Vision project in Azure Portal
2. Train an object detection model
3. Publish the iteration
4. Get the prediction URL and key from the Prediction tab

### Google AutoML Setup

1. Create a Google Cloud Project
2. Enable AutoML API
3. Create and train an object detection model
4. Deploy the model to an endpoint
5. Set up authentication (service account key or application default credentials)

## Usage

1. **Start the application**:
   ```bash
   streamlit run main.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Configure services** in the sidebar:
   - Select which detection services to use
   - Adjust confidence threshold
   - Configure display options

4. **Upload an image** using the file uploader

5. **Run detection** by clicking the "Run Detection" button

6. **Analyze results** in the different tabs:
   - **Images**: View original and processed images with bounding boxes
   - **Analytics**: Interactive charts and statistics
   - **Details**: Raw detection data
   - **Export**: Download results as CSV

## API Response Formats

### Azure Custom Vision
```json
{
  "predictions": [
    {
      "tagName": "object_class",
      "probability": 0.95,
      "boundingBox": {
        "left": 0.1,
        "top": 0.2,
        "width": 0.3,
        "height": 0.4
      }
    }
  ]
}
```

### Google AutoML
```json
{
  "payload": [
    {
      "displayName": "object_class",
      "imageObjectDetection": {
        "score": 0.95,
        "boundingBox": {
          "normalizedVertices": [
            {"x": 0.1, "y": 0.2},
            {"x": 0.4, "y": 0.6}
          ]
        }
      }
    }
  ]
}
```

## Logging

The application logs to both console and file (`object_detection_dashboard.log`). Log levels include:

- **INFO**: General application flow
- **WARNING**: Non-critical issues
- **ERROR**: Error conditions with stack traces

## Troubleshooting

### Common Issues

1. **"No detection services are configured"**
   - Check your `.env` file
   - Verify environment variables are set correctly
   - Ensure service credentials are valid

2. **Azure detection fails**
   - Verify prediction URL and key
   - Check if the model iteration is published
   - Ensure the image format is supported

3. **Google detection fails**
   - Verify Google Cloud credentials
   - Check if the endpoint is deployed
   - Ensure AutoML API is enabled

4. **Import errors**
   - Activate your virtual environment
   - Install all requirements: `pip install -r requirements.txt`

### Performance Optimization

- Resize large images before upload
- Use appropriate confidence thresholds
- Consider running services individually for better error isolation

## Development

### Adding New Services

1. Implement the `DetectionService` interface in `src/models/detection_models.py`
2. Create a new service class in `src/services/`
3. Register the service in `ServiceFactory`
4. Update configuration management

### Testing

While automated tests are not included, you can test the application by:

1. Using sample images from different domains
2. Testing with various confidence thresholds
3. Verifying both services work independently
4. Checking error handling with invalid inputs

## License

This project is provided as-is for educational and development purposes.

## Support

For issues and questions:
1. Check the application logs
2. Verify your configuration
3. Ensure all dependencies are installed
4. Review the troubleshooting section above
