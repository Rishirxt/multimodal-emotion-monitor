"""
run_camera.py  —  entry-point for the multimodal emotion monitor.

Usage:
    python run_camera.py --face Models/strong_face_net.pt --emotion Models/emotion_model.pt

Controls:
    Q / ESC  →  quit
"""
from WebcamEmotion import main

if __name__ == '__main__':
    main()
