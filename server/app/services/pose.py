from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any
import logging
import os

import cv2
import numpy as np
from ultralytics import YOLO


logger = logging.getLogger(__name__)
OPENVINO_EXPORT_IMGSZ = 480

KEYPOINT_NAMES = [
    ('nose', 'head'),
    ('left_eye', 'head'),
    ('right_eye', 'head'),
    ('left_ear', 'head'),
    ('right_ear', 'head'),
    ('left_shoulder', 'upper body'),
    ('right_shoulder', 'upper body'),
    ('left_elbow', 'arm'),
    ('right_elbow', 'arm'),
    ('left_wrist', 'forearm/hand'),
    ('right_wrist', 'forearm/hand'),
    ('left_hip', 'torso/pelvis'),
    ('right_hip', 'torso/pelvis'),
    ('left_knee', 'leg'),
    ('right_knee', 'leg'),
    ('left_ankle', 'foot/ankle'),
    ('right_ankle', 'foot/ankle'),
]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_yolo_weights() -> Path | None:
    candidate_paths: list[Path] = []

    env_value = os.getenv('POSE_BENCHMARK_YOLO_WEIGHTS')
    if env_value:
        candidate_paths.append(Path(env_value).expanduser())

    root = _workspace_root()
    candidate_paths.extend([
        root / 'legacy' / 'legacy_website' / 'yolo11n-pose.pt',
        root / 'server' / 'models' / 'yolo11n-pose.pt',
        root / 'benchmark' / 'weights' / 'yolo11n-pose.pt',
    ])

    for path in candidate_paths:
        if path.exists():
            return path

    return None


def _resolve_yolo_openvino_model() -> Path | None:
    candidate_paths: list[Path] = []

    env_value = os.getenv('POSE_BENCHMARK_YOLO_OPENVINO_MODEL')
    if env_value:
        candidate_paths.append(Path(env_value).expanduser())

    root = _workspace_root()
    candidate_paths.extend([
        root / 'legacy' / 'legacy_website' / f'yolo11n-pose_openvino_{OPENVINO_EXPORT_IMGSZ}_model',
        root / 'server' / 'models' / f'yolo11n-pose_openvino_{OPENVINO_EXPORT_IMGSZ}_model',
        root / 'benchmark' / 'weights' / f'yolo11n-pose_openvino_{OPENVINO_EXPORT_IMGSZ}_model',
        root / 'legacy' / 'legacy_website' / 'yolo11n-pose_openvino_model',
        root / 'server' / 'models' / 'yolo11n-pose_openvino_model',
        root / 'benchmark' / 'weights' / 'yolo11n-pose_openvino_model',
    ])

    for path in candidate_paths:
        if path.exists():
            return path

    return None


def _export_yolo_openvino_model(pt_weights: Path) -> Path:
    export_dir = pt_weights.with_name(f'{pt_weights.stem}_openvino_{OPENVINO_EXPORT_IMGSZ}_model')

    if export_dir.exists():
        return export_dir

    logger.info('Exporting YOLO pose weights to OpenVINO: %s', pt_weights)
    model = YOLO(str(pt_weights))
    exported_model = model.export(format='openvino', imgsz=OPENVINO_EXPORT_IMGSZ)

    if exported_model is not None:
        exported_path = Path(str(exported_model))
        if exported_path.exists():
            return exported_path

    if export_dir.exists():
        return export_dir

    raise FileNotFoundError(f'OpenVINO export did not create a usable model at {export_dir}')


@dataclass
class PoseStats:
    frames: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0
    error: str | None = None
    capture_fps: float = 0.0
    process_fps: float = 0.0
    latest_keypoints: list[dict[str, Any]] = field(default_factory=list)

    def record(self, elapsed_ms: float) -> None:
        self.frames += 1
        self.total_ms += elapsed_ms
        self.last_ms = elapsed_ms
        elapsed_seconds = self.total_ms / 1000.0
        self.process_fps = self.frames / elapsed_seconds if elapsed_seconds > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        average_ms = self.total_ms / self.frames if self.frames else 0.0
        return {
            'frames': self.frames,
            'last_ms': self.last_ms,
            'average_ms': average_ms,
            'capture_fps': self.capture_fps,
            'process_fps': self.process_fps,
            'error': self.error,
            'keypoints': self.latest_keypoints,
        }


class PoseService:
    def __init__(self) -> None:
        self._status = 'idle'
        self._backend = 'yolov11n-pose'
        self._model: YOLO | None = None
        self._model_path: Path | None = None
        self._model_lock = Lock()
        self._load_lock = Lock()

    @property
    def model_lock(self) -> Lock:
        return self._model_lock

    def snapshot(self) -> dict[str, str]:
        return {
            'status': self._status,
            'backend': self._backend,
            'model_path': str(self._model_path) if self._model_path else '',
        }

    def prime(self) -> None:
        model = self.load_model()
        self._warmup_model(model)

    def load_model(self) -> YOLO:
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            openvino_model = _resolve_yolo_openvino_model()
            if openvino_model is not None:
                logger.info('Loading OpenVINO pose model from %s', openvino_model)
                self._model = YOLO(str(openvino_model))
                self._model_path = openvino_model
                self._backend = 'yolov11n-pose-openvino'
                self._status = 'ready'
                return self._model

            pt_weights = _resolve_yolo_weights()
            if pt_weights is None:
                raise FileNotFoundError('YOLO weights not found for pose estimation')

            try:
                openvino_model = _export_yolo_openvino_model(pt_weights)
                logger.info('Loading exported OpenVINO pose model from %s', openvino_model)
                self._model = YOLO(str(openvino_model))
                self._model_path = openvino_model
                self._backend = 'yolov11n-pose-openvino'
            except Exception as exc:  # pragma: no cover - export depends on environment
                logger.warning('OpenVINO export unavailable, loading PT weights directly: %s', exc)
                self._model = YOLO(str(pt_weights))
                self._model_path = pt_weights
                self._backend = 'yolov11n-pose'

            self._status = 'ready'
            return self._model

    def predict(self, frame: np.ndarray):
        model = self.load_model()
        with self._model_lock:
            return model.predict(frame, verbose=False)

    def annotate_frame(self, frame: np.ndarray, results: list[Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        annotated = frame.copy()
        detected_keypoints: list[dict[str, Any]] = []

        if not results:
            return annotated, detected_keypoints

        result = results[0]
        keypoints = getattr(result, 'keypoints', None)
        if keypoints is None or keypoints.xy is None:
            return annotated, detected_keypoints

        xy = keypoints.xy.cpu().numpy()
        conf = keypoints.conf.cpu().numpy() if getattr(keypoints, 'conf', None) is not None else None
        height, width = annotated.shape[:2]
        colors = [
            (255, 99, 71),
            (255, 165, 0),
            (255, 215, 0),
            (50, 205, 50),
            (64, 224, 208),
            (30, 144, 255),
        ]

        for person_index, person_keypoints in enumerate(xy):
            person_conf = conf[person_index] if conf is not None else None

            for index, (name, region) in enumerate(KEYPOINT_NAMES):
                if index >= len(person_keypoints):
                    break

                x_value, y_value = person_keypoints[index]
                confidence = float(person_conf[index]) if person_conf is not None else None

                if x_value <= 0 or y_value <= 0:
                    continue

                x_int = int(round(x_value))
                y_int = int(round(y_value))
                color = colors[index % len(colors)]

                cv2.circle(annotated, (x_int, y_int), 4, color, -1, cv2.LINE_AA)

                detected_keypoints.append({
                    'person_index': person_index,
                    'index': index,
                    'name': name,
                    'represents': region,
                    'x': float(round(x_value / width, 4)),
                    'y': float(round(y_value / height, 4)),
                    'confidence': float(round(confidence, 4)) if confidence is not None else None,
                })

        return annotated, detected_keypoints

    def _warmup_model(self, model: YOLO, iterations: int = 2) -> None:
        warmup_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for _ in range(max(0, iterations)):
            try:
                model.predict(warmup_frame, verbose=False)
            except Exception as exc:  # pragma: no cover - warmup is best-effort only
                logger.warning('YOLO warmup skipped after failure: %s', exc)
                break


pose_service = PoseService()
