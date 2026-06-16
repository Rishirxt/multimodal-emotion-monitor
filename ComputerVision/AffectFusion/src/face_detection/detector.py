import os
import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self._backend = None
        self.face_detection = None
        self.face_cascade = None
        self._mp_image_cls = None
        self._mp_image_format = None

        if hasattr(mp, 'solutions'):
            self._init_mediapipe_solutions(min_detection_confidence)
        elif hasattr(mp, 'tasks'):
            self._init_mediapipe_tasks(min_detection_confidence)
        else:
            self._init_opencv_cascade()

    def _init_mediapipe_solutions(self, min_detection_confidence):
        self._backend = 'mediapipe_solutions'
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def _init_mediapipe_tasks(self, min_detection_confidence):
        model_path = self._find_mediapipe_model()
        if model_path is None:
            self._init_opencv_cascade()
            return

        try:
            from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
        except ImportError:
            self._init_opencv_cascade()
            return

        self._backend = 'mediapipe_tasks'
        self._mp_image_cls = Image
        self._mp_image_format = ImageFormat
        self.face_detection = mp.tasks.vision.FaceDetector.create_from_model_path(
            model_path
        )

    def _init_opencv_cascade(self):
        self._backend = 'opencv'
        cascade_path = os.path.join(
            cv2.data.haarcascades,
            'haarcascade_frontalface_default.xml',
        )
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(
                f'Failed to load OpenCV cascade classifier from {cascade_path}'
            )

    def _find_mediapipe_model(self):
        if not hasattr(mp, 'tasks'):
            return None

        candidates = [
            'face_detection_short_range.tflite',
            'face_detection_full_range.tflite',
        ]
        search_paths = [
            os.getcwd(),
            os.path.dirname(mp.__file__),
        ]

        for candidate in candidates:
            for base in search_paths:
                path = os.path.join(base, candidate)
                if os.path.isfile(path):
                    return path

        return None

    def detect(self, frame):
        """
        Detect faces in frame.

        Returns:
            List of tuples:
            [(x, y, w, h), ...]
        """
        if self._backend == 'mediapipe_solutions':
            return self._detect_mediapipe_solutions(frame)
        if self._backend == 'mediapipe_tasks':
            return self._detect_mediapipe_tasks(frame)
        return self._detect_opencv(frame)

    def _detect_mediapipe_solutions(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                faces.append((x, y, width, height))
        return faces

    def _detect_mediapipe_tasks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image_obj = self._mp_image_cls(
            image_format=self._mp_image_format.SRGB,
            data=rgb_frame,
        )

        results = self.face_detection.detect(mp_image_obj)

        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                faces.append((x, y, width, height))
        return faces

    def _detect_opencv(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        return [tuple(rect) for rect in detections]

    def close(self):
        if self._backend in ('mediapipe_solutions', 'mediapipe_tasks') and self.face_detection is not None:
            self.face_detection.close()
