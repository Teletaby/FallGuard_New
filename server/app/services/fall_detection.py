from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import time
from typing import Any
import logging
import os
import uuid

import numpy as np
import torch
import torch.nn as nn

from app.state import state


logger = logging.getLogger(__name__)
MODEL_FILENAME = 'fall_detection.pt'
SEQUENCE_LENGTH = 24
FRAME_STRIDE = 4
FEATURE_SIZE = 51
PERSON_IOU_THRESHOLD = 0.25
TRACK_TTL_FRAMES = 60
ALERT_HOLD_SECONDS = 8.0


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_model_path() -> Path | None:
    env_value = os.getenv('POSE_FALL_MODEL_PATH')
    if env_value:
        candidate = Path(env_value).expanduser()
        if candidate.exists():
            return candidate

    root = _workspace_root()
    candidate_paths = [
        root / 'models' / MODEL_FILENAME,
        root / 'legacy_server' / 'models' / MODEL_FILENAME,
        root / 'benchmark' / 'weights' / MODEL_FILENAME,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return None


class PoseLSTM(nn.Module):
    def __init__(self, input_size: int = FEATURE_SIZE, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 2) -> None:
        super().__init__()
        self.pre_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_proj(x)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


@dataclass
class PersonTrack:
    track_id: str
    label: int
    last_box: tuple[float, float, float, float]
    last_seen_frame_index: int
    last_sampled_frame_index: int = -FRAME_STRIDE
    features: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=SEQUENCE_LENGTH))
    fall_active: bool = False
    last_fall_probability: float = 0.0


class FallDetectionService:
    def __init__(self, alert_service: Any, incident_service: Any) -> None:
        self._alert_service = alert_service
        self._incident_service = incident_service
        self._model: PoseLSTM | None = None
        self._model_path: Path | None = None
        self._device = torch.device('cpu')
        self._model_lock = Lock()
        self._track_lock = Lock()
        self._tracks: dict[str, dict[str, PersonTrack]] = {}
        self._next_labels: dict[str, int] = {}
        self._camera_hold_until: dict[str, float] = {}
        self._camera_loop_epochs: dict[str, int] = {}

    def load_model(self) -> PoseLSTM:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            model_path = _resolve_model_path()
            if model_path is None:
                raise FileNotFoundError('fall detection checkpoint not found')

            model = PoseLSTM()
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            model.to(self._device)
            model.eval()

            self._model = model
            self._model_path = model_path
            logger.info('Loaded fall detection checkpoint from %s', model_path)
            return self._model

    def process_frame(self, camera_id: str, camera_name: str, frame_index: int, frame: np.ndarray, results: list[Any], loop_epoch: int) -> list[dict[str, Any]]:
        if not self._is_current_epoch(camera_id, loop_epoch):
            return []

        if not results:
            self._sync_camera_state(camera_id)
            self._prune_tracks(camera_id, frame_index)
            return []

        result = results[0]
        boxes = getattr(result, 'boxes', None)
        keypoints = getattr(result, 'keypoints', None)
        if boxes is None or getattr(boxes, 'xyxy', None) is None or keypoints is None:
            self._sync_camera_state(camera_id)
            self._prune_tracks(camera_id, frame_index)
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        keypoints_xy = getattr(keypoints, 'xyn', None)
        if keypoints_xy is None:
            keypoints_xy = getattr(keypoints, 'xy', None)
        if keypoints_xy is None:
            self._sync_camera_state(camera_id)
            self._prune_tracks(camera_id, frame_index)
            return []

        if hasattr(keypoints_xy, 'cpu'):
            keypoints_xy = keypoints_xy.cpu().numpy()
        else:
            keypoints_xy = np.asarray(keypoints_xy)

        detections = min(len(xyxy), len(keypoints_xy))
        emitted_alerts: list[dict[str, Any]] = []

        with self._track_lock:
            tracks = self._tracks.setdefault(camera_id, {})

            matched_track_ids: set[str] = set()
            for detection_index in range(detections):
                box = tuple(float(value) for value in xyxy[detection_index][:4])
                person_keypoints = keypoints_xy[detection_index]

                if not self._has_visible_keypoints(person_keypoints):
                    continue

                track = self._match_track(camera_id, box, frame_index)
                matched_track_ids.add(track.track_id)

                if self._should_sample(track, frame_index):
                    features = self._build_features(person_keypoints, frame.shape)
                    track.features.append(features)
                    track.last_sampled_frame_index = frame_index

                    if not self._is_current_epoch(camera_id, loop_epoch):
                        return emitted_alerts

                    fall_probability, is_fall = self._predict_track(track)
                    track.last_fall_probability = fall_probability
                    is_fall = fall_probability >= self._fall_threshold()

                    if not self._is_current_epoch(camera_id, loop_epoch):
                        return emitted_alerts

                    if is_fall and not track.fall_active:
                        track.fall_active = True
                        alert = self._alert_service.record_alert(camera_id, camera_name, fall_probability, person_id=track.label)
                        emitted_alerts.append(alert)
                        self._incident_service.create_incident(
                            {
                                'camera_id': camera_id,
                                'camera_name': camera_name,
                                'confidence': fall_probability,
                                'person_id': track.label,
                                'person_label': f'Person #{track.label}',
                            },
                            frame=frame,
                        )
                        self._camera_hold_until[camera_id] = time() + ALERT_HOLD_SECONDS
                    elif not is_fall:
                        track.fall_active = False

                track.last_box = box
                track.last_seen_frame_index = frame_index

            for track_id in list(tracks.keys()):
                if track_id not in matched_track_ids and frame_index - tracks[track_id].last_seen_frame_index > TRACK_TTL_FRAMES:
                    tracks.pop(track_id, None)

        self._sync_camera_state(camera_id)
        return emitted_alerts

    def get_track_annotations(self, camera_id: str) -> list[dict[str, Any]]:
        with self._track_lock:
            tracks = list(self._tracks.get(camera_id, {}).values())

        annotations: list[dict[str, Any]] = []
        for track in tracks:
            annotations.append({
                'label': f'Person #{track.label}',
                'confidence': float(track.last_fall_probability),
                'fall_active': bool(track.fall_active),
                'box': tuple(float(value) for value in track.last_box),
            })

        return annotations

    def reset_camera(self, camera_id: str) -> None:
        self._camera_loop_epochs[camera_id] = self._camera_loop_epochs.get(camera_id, 0) + 1

        with self._track_lock:
            self._tracks.pop(camera_id, None)
            self._camera_hold_until.pop(camera_id, None)
            self._next_labels.pop(camera_id, None)

        with state.lock:
            camera = state.cameras.get(camera_id)
            if camera is not None:
                camera['color'] = 'green'
                camera['confidence_score'] = 0.0

    def _is_current_epoch(self, camera_id: str, loop_epoch: int) -> bool:
        return self._camera_loop_epochs.get(camera_id, 0) == loop_epoch

    def _match_track(self, camera_id: str, box: tuple[float, float, float, float], frame_index: int) -> PersonTrack:
        tracks = self._tracks.setdefault(camera_id, {})
        best_track: PersonTrack | None = None
        best_iou = 0.0

        for track in tracks.values():
            if frame_index - track.last_seen_frame_index > TRACK_TTL_FRAMES:
                continue

            iou = self._intersection_over_union(box, track.last_box)
            if iou > best_iou:
                best_iou = iou
                best_track = track

        if best_track is not None and best_iou >= PERSON_IOU_THRESHOLD:
            return best_track

        next_label = self._next_labels.get(camera_id, 0) + 1
        self._next_labels[camera_id] = next_label
        track = PersonTrack(
            track_id=uuid.uuid4().hex,
            label=next_label,
            last_box=box,
            last_seen_frame_index=frame_index,
        )
        tracks[track.track_id] = track
        return track

    @staticmethod
    def _should_sample(track: PersonTrack, frame_index: int) -> bool:
        return not track.features or frame_index - track.last_sampled_frame_index >= FRAME_STRIDE

    def _predict_track(self, track: PersonTrack) -> tuple[float, bool]:
        model = self.load_model()

        sequence = list(track.features)
        if not sequence:
            sequence = [np.zeros(FEATURE_SIZE, dtype=np.float32)]

        while len(sequence) < SEQUENCE_LENGTH:
            sequence.append(sequence[-1].copy())

        window = np.stack(sequence[-SEQUENCE_LENGTH:], axis=0).astype(np.float32)
        input_tensor = torch.from_numpy(window).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            fall_probability = float(probabilities[0, 1].item())
            prediction = int(torch.argmax(probabilities, dim=-1).item())

        return fall_probability, prediction == 1

    def _build_features(self, person_keypoints: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
        features = np.zeros((17, 3), dtype=np.float32)
        max_joints = min(17, len(person_keypoints))
        height = float(frame_shape[0]) if len(frame_shape) >= 1 and frame_shape[0] else 1.0
        width = float(frame_shape[1]) if len(frame_shape) >= 2 and frame_shape[1] else 1.0
        values = np.asarray(person_keypoints, dtype=np.float32)
        normalize_pixels = bool(np.nanmax(values[:, :2]) > 1.5) if values.size else False

        for joint_index in range(max_joints):
            joint = values[joint_index]
            if len(joint) >= 2:
                x_value = float(joint[0])
                y_value = float(joint[1])
                if normalize_pixels:
                    x_value /= width
                    y_value /= height

                features[joint_index, 0] = x_value
                features[joint_index, 1] = y_value

        if not np.any(features[:, :2]):
            return features.reshape(-1)

        origin_x = features[0, 0]
        origin_y = features[0, 1]
        features[:, 0] -= origin_x
        features[:, 1] -= origin_y

        norm = float(np.linalg.norm(features[:, :2].reshape(-1)))
        if norm > 0:
            features[:, :2] /= norm

        return features.reshape(-1)

    @staticmethod
    def _has_visible_keypoints(person_keypoints: np.ndarray) -> bool:
        values = np.asarray(person_keypoints, dtype=np.float32)
        if values.size == 0 or values.shape[0] == 0:
            return False

        coordinates = values[:, :2]
        if not np.isfinite(coordinates).any():
            return False

        visible_points = np.isfinite(coordinates).all(axis=1) & (coordinates[:, 0] > 0.0) & (coordinates[:, 1] > 0.0)
        return bool(np.any(visible_points))

    @staticmethod
    def _intersection_over_union(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(0.0, intersection_x2 - intersection_x1)
        intersection_height = max(0.0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def _prune_tracks(self, camera_id: str, frame_index: int) -> None:
        with self._track_lock:
            tracks = self._tracks.get(camera_id)
            if not tracks:
                return

            for track_id in list(tracks.keys()):
                if frame_index - tracks[track_id].last_seen_frame_index > TRACK_TTL_FRAMES:
                    tracks.pop(track_id, None)

    def _sync_camera_state(self, camera_id: str) -> None:
        now = time()
        with self._track_lock:
            tracks = list(self._tracks.get(camera_id, {}).values())
            hold_until = self._camera_hold_until.get(camera_id, 0.0)

        active_tracks = [track for track in tracks if track.fall_active]
        camera_is_falling = bool(active_tracks) or now < hold_until
        confidence = max((track.last_fall_probability for track in active_tracks), default=0.0) if camera_is_falling else 0.0

        with state.lock:
            camera = state.cameras.get(camera_id)
            if camera is not None:
                camera['color'] = 'red' if camera_is_falling else 'green'
                camera['confidence_score'] = float(confidence)

    def _fall_threshold(self) -> float:
        with state.lock:
            value = state.detection_settings.get('fall_threshold', 0.8)

        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return 0.8

        if threshold < 0.0:
            return 0.0
        if threshold > 1.0:
            return 1.0
        return threshold
