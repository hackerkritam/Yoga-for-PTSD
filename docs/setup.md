# Setup Guide - Yoga PTSD System

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB free space
- **Webcam**: Built-in or external USB camera

### Hardware Requirements
- **Camera**: Any webcam with 640x480 resolution or higher
- **Display**: 1280x720 minimum resolution (for side-by-side view)
- **Processing**: Intel i3 or equivalent (for real-time processing)

## Installation Steps

### 1. Python Installation
If you don't have Python installed:
1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. Install with "Add to PATH" option checked
3. Verify installation: `python --version`

### 2. Project Setup
```bash
# Clone or download the project
cd Yoga_PTSD_System

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Test MediaPipe installation
python -c "import mediapipe as mp; print('MediaPipe installed successfully')"

# Test OpenCV installation
python -c "import cv2; print('OpenCV installed successfully')"
```

## Running the Application

### Basic Run
```bash
python src/main.py
```

### Troubleshooting

#### Common Issues:

1. **Camera not found**
   - Check if webcam is connected
   - Try different camera index: `cv2.VideoCapture(1)`

2. **MediaPipe not working**
   - Update MediaPipe: `pip install --upgrade mediapipe`
   - Check internet connection (first run downloads models)

3. **Performance issues**
   - Close other applications
   - Reduce camera resolution in code
   - Lower detection confidence threshold

4. **Import errors**
   - Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
   - Check Python version compatibility

## Configuration

### Camera Settings
- **Default Resolution**: 640x480
- **Frame Rate**: 30 FPS
- **Camera Index**: 0 (built-in camera)

### Pose Detection Settings
- **Detection Confidence**: 70%
- **Tracking Confidence**: 70%
- **Landmark Points**: 33 body points

### Timing Settings
- **Mountain Pose**: 2 seconds
- **Tree Pose**: 3 seconds
- **Warrior Pose**: 2.5 seconds
- **Child's Pose**: 2 seconds
- **Lotus Pose**: 3 seconds

## Performance Optimization

### For Better Performance:
1. **Close background applications**
2. **Use dedicated graphics card** (if available)
3. **Ensure good lighting** for better pose detection
4. **Stand 6-8 feet from camera** for optimal detection

### For Lower-end Systems:
1. **Reduce camera resolution** in code
2. **Lower detection confidence** to 50%
3. **Disable unnecessary features**

## Security and Privacy

### Data Handling:
- **No data is stored** - all processing is real-time
- **No internet connection required** after initial setup
- **Camera feed is not recorded** or saved

### Privacy Features:
- **Local processing only**
- **No cloud services used**
- **No personal data collected**

## Support

### Getting Help:
1. Check this setup guide
2. Review the README.md file
3. Check system requirements
4. Verify Python and dependency versions

### Contact:
For technical support, refer to the project documentation or create an issue in the project repository. 