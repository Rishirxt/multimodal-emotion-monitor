import cv2
import time


class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()

    def get_fps(self):
        current_time = time.time()

        fps = 1 / (current_time - self.prev_time)

        self.prev_time = current_time

        return int(fps)


def draw_face_boxes(frame, faces):
    """
    Draw bounding boxes around detected faces.
    """

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame


def draw_fps(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    return frame