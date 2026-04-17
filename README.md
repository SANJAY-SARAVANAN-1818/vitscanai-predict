# VitScanAI - Live Camera Predict 🔍

A modern web application for real-time camera feed capture and AI-powered predictions with an intuitive dashboard interface.

## Features ✨

- **Live Camera Feed**: Real-time video streaming from your webcam
- **Quick Action Buttons**: One-click controls for camera operations
- **Image Capture**: Screenshot functionality with timestamps
- **Video Recording**: Toggle recording on/off
- **Statistics Dashboard**: Frame count and session duration tracking
- **Modern UI**: Dark theme with responsive design
- **OpenCV Integration**: Advanced image processing capabilities
- **Real-time Status**: Live camera and recording status indicators

## Quick Actions 🚀

- **▶ Start Camera**: Initialize and start the camera feed
- **⏹ Stop Camera**: Safely stop camera and release resources
- **📸 Capture**: Take a screenshot from current frame
- **🔴 Record**: Toggle video recording on/off

## Installation 📦

### Prerequisites
- Python 3.7+
- Webcam/Camera device

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SANJAY-SARAVANAN-1818/vitscanai-predict.git
   cd vitscanai-predict
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and navigate to: `http://localhost:5000`

## Project Structure 📁

```
vitscanai-predict/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── style.css         # Stylesheet
│   └── script.js         # JavaScript functionality
└── uploads/              # Captured images storage
```

## API Endpoints 🔌

### Camera Control

- **POST** `/api/camera/start` - Start camera stream
- **POST** `/api/camera/stop` - Stop camera stream
- **POST** `/api/camera/capture` - Capture single frame
- **POST** `/api/camera/toggle-recording` - Toggle recording
- **GET** `/api/camera/status` - Get camera status

### Streaming

- **GET** `/video_feed` - Live video stream (MJPEG)
- **GET** `/` - Main dashboard page

## Features Breakdown 🎯

### Camera Handler Class
- Thread-safe camera operations
- Frame encoding and processing
- Timestamp overlay on frames
- Resource management

### Web Interface
- Responsive grid layout
- Real-time status indicators
- Settings panel for adjustments
- Captures history panel
- Keyboard shortcuts support

### Keyboard Shortcuts ⌨️

- **Space**: Capture image
- **Ctrl+S**: Start camera

## Settings ⚙️

Adjust camera parameters in the dashboard:
- **Brightness**: 0-100%
- **Contrast**: 0-100%
- **AI Prediction**: Enable/Disable

## Camera Specifications

- **Resolution**: 1280x720 (HD)
- **FPS**: 30 frames per second
- **Format**: MJPEG streaming

## Troubleshooting 🔧

### Camera not detected
- Ensure camera is connected and not in use by other applications
- Check permissions (Linux): `ls -la /dev/video0`

### Port already in use
```bash
# Change port in app.py: app.run(port=5001)
# Or kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

### Performance issues
- Reduce resolution or FPS in app.py
- Close unnecessary applications
- Check system resources

## Performance Tips 🚀

1. Close other camera applications
2. Ensure adequate lighting for better video quality
3. Use a USB 3.0 camera for better performance
4. Run on a system with sufficient RAM

## API Response Examples 📊

### Start Camera
```json
{
  "status": "success",
  "message": "Camera started"
}
```

### Camera Status
```json
{
  "status": "success",
  "camera_active": true,
  "is_recording": false,
  "frame_count": 150
}
```

### Capture Image
```json
{
  "status": "success",
  "filename": "capture_20260417_153000.jpg",
  "path": "uploads/capture_20260417_153000.jpg"
}
```

## Browser Compatibility 🌐

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Security Considerations 🔒

- Application runs on localhost by default
- Camera access is browser/OS controlled
- Images stored in uploads folder
- No data sent to external servers (by default)

## Future Enhancements 🔮

- AI model integration for predictions
- Multi-camera support
- Video export functionality
- Advanced image filters
- Cloud storage integration
- Real-time analytics

## License 📜

This project is open source and available under the MIT License.

## Contributing 🤝

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support & Contact 📧

For issues and questions:
- GitHub Issues: [Report Issues](https://github.com/SANJAY-SARAVANAN-1818/vitscanai-predict/issues)
- Email: contact@vitscanai.com

---

**Made with ❤️ by VitScanAI Team**

Last Updated: 2026-04-17 05:52:13