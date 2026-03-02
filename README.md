# Fox Devil Gesture Recognition 🦊

A real-time AI gesture recognition system based on the popular anime **Chainsaw Man**. When the user makes the signature 'Fox Devil' (Kon) hand gesture, the system detects it using the webcam and summons the Fox Devil with dynamic 3D visual and sound effects.

## ✨ Features
- **Real-time Gesture Recognition:** Uses MediaPipe Hand Tracking to accurately detect the "Kon" hand sign.
- **Dynamic 3D Visual Effects:** 
  - Real 3D video/image rendering of the Fox Devil bite.
  - Aggressive jump-scare zoom animation.
  - Screen shake, blood-red blackout, and particle effects overlay.
  - Chroma-key processing for video assets (automatically removes green screen).
- **Custom Sound Effects:** Automatically plays the custom "Kon" sound followed by an attack sound.
- **Face/Hand Tracking:** The Fox Devil targets and attacks the user's face (or center of the hand if face is not visible).

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Webcam

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YunhuPark/fox-devil.git
   cd fox-devil
   ```
2. Install the required Python packages:
   ```bash
   pip install opencv-python mediapipe numpy pygame imageio-ffmpeg
   ```

### Usage
Run the main script:
```bash
python fox_devil_gesture.py
```
- Show the "Kon" gesture (index & pinky extended, middle & ring bent, thumb touching middle/ring) to the webcam.
- Press `q` to quit the application.

## 📁 File Structure
- `fox_devil_gesture.py` - Main application logic.
- `sounds/` - Directory containing your media assets (`fox_head.png`, `user_sound.wav`, and the `.mp4` 3D video effect).
- `hand_landmarker.task` - MediaPipe Hand Tracking model.
- `blaze_face_short_range.tflite` - MediaPipe Face Detection model.

## 🛠️ Built With
- [OpenCV](https://opencv.org/) - Real-time computer vision and rendering.
- [MediaPipe](https://developers.google.com/mediapipe) - Hand landmark tracking and face detection.
- [Pygame](https://www.pygame.org/) - Audio playback.

## 📄 License
This project is open-source and available under the MIT License.
