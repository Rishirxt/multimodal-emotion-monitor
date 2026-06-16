import cv2

from face_detection.detector import FaceDetector
from face_detection.utils import draw_face_boxes
from face_detection.utils import draw_fps
from face_detection.utils import FPSCounter


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = FaceDetector()
    fps_counter = FPSCounter()

    print("Starting webcam...")
    print("Press 'q' to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)

        faces = detector.detect(frame)

        frame = draw_face_boxes(frame, faces)

        fps = fps_counter.get_fps()
        frame = draw_fps(frame, fps)

        cv2.putText(
            frame,
            f"Faces Detected: {len(faces)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow("AffectFusion - Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()