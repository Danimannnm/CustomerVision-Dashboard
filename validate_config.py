"""
Configuration validation script for the Object Detection Dashboard.
Run this script to check if your environment is properly configured.
"""
import os
import sys
from dotenv import load_dotenv

def check_environment_variables():
    """Check if required environment variables are set."""
    print("🔧 Checking Environment Variables...")
    
    load_dotenv()
    
    # Azure Custom Vision
    azure_url = os.getenv('CUSTOMVISION_PREDICTION_URL')
    azure_key = os.getenv('CUSTOMVISION_PREDICTION_KEY')
    
    # Google AutoML
    google_project = os.getenv('GOOGLE_PROJECT_ID')
    google_endpoint = os.getenv('GOOGLE_ENDPOINT_ID')
    google_location = os.getenv('GOOGLE_LOCATION')
    
    print("\n📋 Azure Custom Vision:")
    if azure_url and azure_key:
        print("  ✅ CUSTOMVISION_PREDICTION_URL: Set")
        print("  ✅ CUSTOMVISION_PREDICTION_KEY: Set")
    else:
        print("  ❌ Azure Custom Vision not configured")
        print("     Missing: CUSTOMVISION_PREDICTION_URL or CUSTOMVISION_PREDICTION_KEY")
    
    print("\n📋 Google AutoML:")
    if google_project and google_endpoint:
        print("  ✅ GOOGLE_PROJECT_ID: Set")
        print("  ✅ GOOGLE_ENDPOINT_ID: Set")
        print(f"  ✅ GOOGLE_LOCATION: {google_location or 'us-central1 (default)'}")
    else:
        print("  ❌ Google AutoML not configured")
        print("     Missing: GOOGLE_PROJECT_ID or GOOGLE_ENDPOINT_ID")
    
    return bool(azure_url and azure_key), bool(google_project and google_endpoint)

def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        'streamlit',
        'python-dotenv',
        'PIL',
        'cv2',
        'numpy',
        'pandas',
        'plotly',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'python-dotenv':
                import dotenv
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_google_credentials():
    """Check Google Cloud credentials."""
    print("\n🔐 Checking Google Cloud Credentials...")
    
    try:
        from google.auth import default
        credentials, project = default()
        print("  ✅ Google Cloud credentials found")
        print(f"  📋 Project: {project}")
        return True
    except Exception as e:
        print("  ❌ Google Cloud credentials not found")
        print(f"     Error: {e}")
        print("💡 Set up authentication: https://cloud.google.com/docs/authentication")
        return False

def main():
    """Main validation function."""
    print("🔍 Object Detection Dashboard - Configuration Validator")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check environment variables
    azure_ok, google_ok = check_environment_variables()
    
    # Check Google credentials (only if Google is configured)
    google_creds_ok = True
    if google_ok:
        google_creds_ok = check_google_credentials()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Configuration Summary:")
    print(f"  Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"  Azure Custom Vision: {'✅' if azure_ok else '❌'}")
    print(f"  Google AutoML Config: {'✅' if google_ok else '❌'}")
    print(f"  Google Credentials: {'✅' if google_creds_ok else '❌'}")
    
    if deps_ok and (azure_ok or (google_ok and google_creds_ok)):
        print("\n🎉 Configuration looks good! You can run the dashboard.")
        print("💡 Run: streamlit run main.py")
    else:
        print("\n⚠️  Configuration issues found. Please fix the above issues before running the dashboard.")
    
    if not azure_ok and not (google_ok and google_creds_ok):
        print("\n❌ No services are configured. Please set up at least one service.")

if __name__ == "__main__":
    main()
