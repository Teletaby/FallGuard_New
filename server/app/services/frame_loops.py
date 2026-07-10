from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter, sleep
import time
from typing import Any

import cv2
import numpy as np

from app.services import fall_detection_service
from app.services.pose import PoseStats, pose_service
from app.state import state


@dataclass
class CameraRuntime:
    camera_id: str
    source: str
    source_kind: str
    playback_mode: str
    source_value: Any
    loop_on_eof: bool
    stop_event: Event = field(default_factory=Event)
    finished_event: Event = field(default_factory=Event)
    capture_lock: Lock = field(default_factory=Lock)
    frame_lock: Lock = field(default_factory=Lock)
    pose_queue: Queue = field(default_factory=lambda: Queue(maxsize=4))
    active_capture: Any | None = None
    latest_frame: np.ndarray | None = None
    latest_annotated_frame: np.ndarray | None = None
    latest_frame_jpeg: bytes | None = None
    pose_stats: PoseStats = field(default_factory=PoseStats)
    source_fps: float = 30.0
    target_process_fps: float = 24.0
    frame_index: int = 0
    loop_epoch: int = 0


class FrameLoopService:
    def __init__(self) -> None:
        self._runtimes: dict[str, CameraRuntime] = {}
        self._workers: dict[str, tuple[Thread, Thread]] = {}

    def start(self, camera_record: dict[str, Any]) -> None:
        camera_id = str(camera_record['id'])
        source = str(camera_record.get('source', '')).strip()
        if not source:
            return

        self.stop(camera_id)

        source_kind = str(camera_record.get('source_kind', 'camera'))
        playback_mode = str(camera_record.get('playback_mode', 'live'))
        source_value = self._coerce_capture_source(source, source_kind)

        runtime = CameraRuntime(
            camera_id=camera_id,
            source=source,
            source_kind=source_kind,
            playback_mode=playback_mode,
            source_value=source_value,
            loop_on_eof=source_kind == 'video',
        )

        with state.lock:
            self._runtimes[camera_id] = runtime
            state.frame_sources[camera_id] = {
                'camera_id': camera_id,
                'source': source,
                'source_kind': source_kind,
                'playback_mode': playback_mode,
                'snapshot_url': f'/api/cameras/{camera_id}/snapshot',
                'stream_url': f'/api/cameras/{camera_id}/stream.mjpeg?fps=24',
                'status': 'starting',
                'frame_index': 0,
                'last_frame_at': None,
                'source_fps': runtime.source_fps,
                'target_process_fps': runtime.target_process_fps,
                'pose_process_fps': 0.0,
                'pose_last_ms': 0.0,
                'pose_error': None,
                'latest_keypoints': [],
            }

        reader = Thread(target=self._reader_worker, args=(runtime,), name=f'frame-reader-{camera_id}', daemon=True)
        pose_worker = Thread(target=self._pose_worker, args=(runtime,), name=f'pose-worker-{camera_id}', daemon=True)

        with state.lock:
            self._workers[camera_id] = (reader, pose_worker)

        reader.start()
        pose_worker.start()

    def stop(self, camera_id: str) -> None:
        with state.lock:
            runtime = self._runtimes.pop(camera_id, None)
            self._workers.pop(camera_id, None)

        if runtime is None:
            return

        runtime.stop_event.set()
        self._signal_end_of_stream(runtime)

        with runtime.capture_lock:
            capture = runtime.active_capture
            runtime.active_capture = None

        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    def snapshot(self, camera_id: str) -> dict[str, Any] | None:
        with state.lock:
            source = state.frame_sources.get(camera_id)
            return dict(source) if source else None

    def is_active(self, camera_id: str) -> bool:
        with state.lock:
            runtime = self._runtimes.get(camera_id)
        return runtime is not None and not runtime.stop_event.is_set()

    def stream_frame(self, camera_id: str) -> bytes:
        with state.lock:
            runtime = self._runtimes.get(camera_id)

        if runtime is None:
            return self._placeholder_bytes()

        with runtime.frame_lock:
            if runtime.latest_frame_jpeg:
                return runtime.latest_frame_jpeg

            fallback_frame = runtime.latest_annotated_frame if runtime.latest_annotated_frame is not None else runtime.latest_frame
            if fallback_frame is not None:
                encoded = self._encode_frame(fallback_frame)
                runtime.latest_frame_jpeg = encoded
                return encoded

        return self._placeholder_bytes()

    def _reader_worker(self, runtime: CameraRuntime) -> None:
        if runtime.source_kind == 'webcam':
            capture = cv2.VideoCapture(runtime.source_value)
        else:
            capture = cv2.VideoCapture(str(runtime.source_value))

        try:
            with runtime.capture_lock:
                runtime.active_capture = capture

            if not capture.isOpened():
                self._set_state(runtime.camera_id, status='error', pose_error='unable to open source')
                runtime.stop_event.set()
                self._signal_end_of_stream(runtime)
                return

            source_fps = capture.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps != source_fps or source_fps <= 0:
                source_fps = 30.0

            runtime.source_fps = float(source_fps)
            runtime.target_process_fps = min(24.0, max(1.0, runtime.source_fps))
            frame_interval = 1.0 / max(1.0, runtime.source_fps)
            next_frame_deadline = perf_counter()

            self._set_state(
                runtime.camera_id,
                status='starting',
                source_fps=runtime.source_fps,
                target_process_fps=runtime.target_process_fps,
                pose_error=None,
            )

            while not runtime.stop_event.is_set():
                next_frame_deadline += frame_interval
                success, frame = capture.read()
                if not success:
                    if runtime.loop_on_eof and runtime.source_kind == 'video' and isinstance(runtime.source_value, str) and Path(runtime.source_value).exists():
                        runtime.loop_epoch += 1
                        self._reset_loop_runtime(runtime)
                        fall_detection_service.reset_camera(runtime.camera_id)
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        runtime.frame_index = 0
                        next_frame_deadline = perf_counter()
                        continue

                    self._set_state(runtime.camera_id, status='ended')
                    break

                runtime.frame_index += 1
                with runtime.frame_lock:
                    runtime.latest_frame = frame.copy()

                while not runtime.stop_event.is_set():
                    try:
                        runtime.pose_queue.put_nowait((runtime.loop_epoch, runtime.frame_index, frame.copy()))
                        break
                    except Full:
                        try:
                            runtime.pose_queue.get_nowait()
                        except Empty:
                            break

                self._set_state(
                    runtime.camera_id,
                    status='looping' if runtime.loop_on_eof else 'live',
                    frame_index=runtime.frame_index,
                    last_frame_at=time.time(),
                    source_fps=runtime.source_fps,
                    target_process_fps=runtime.target_process_fps,
                    pose_process_fps=runtime.pose_stats.process_fps,
                    pose_last_ms=runtime.pose_stats.last_ms,
                    pose_error=runtime.pose_stats.error,
                    latest_keypoints=runtime.pose_stats.latest_keypoints,
                )

                sleep_for = next_frame_deadline - perf_counter()
                if sleep_for > 0 and runtime.stop_event.wait(sleep_for):
                    break
                if sleep_for <= 0:
                    next_frame_deadline = perf_counter()
        finally:
            capture.release()
            with runtime.capture_lock:
                runtime.active_capture = None

            self._signal_end_of_stream(runtime)

            with state.lock:
                frame_source = state.frame_sources.get(runtime.camera_id)
                if frame_source is not None and frame_source.get('status') not in {'error', 'ended'}:
                    frame_source['status'] = 'stopped'

            runtime.finished_event.set()

    def _pose_worker(self, runtime: CameraRuntime) -> None:
        self._set_state(runtime.camera_id, status='processing')

        try:
            pose_service.load_model()
        except Exception as exc:
            runtime.pose_stats.error = f'openvino load failed: {exc}'
            self._set_state(runtime.camera_id, status='error', pose_error=runtime.pose_stats.error)
            runtime.stop_event.set()
            self._signal_end_of_stream(runtime)
            return

        while True:
            queue_item = runtime.pose_queue.get()
            if queue_item is None or runtime.stop_event.is_set():
                break

            loop_epoch, frame_index, frame = queue_item
            if loop_epoch != runtime.loop_epoch:
                continue

            start = perf_counter()
            try:
                results = pose_service.predict(frame)
                if loop_epoch != runtime.loop_epoch:
                    continue
                annotated_frame, keypoints = pose_service.annotate_frame(frame, results)
                camera_name = self._camera_name(runtime.camera_id)
                fall_detection_service.process_frame(runtime.camera_id, camera_name, frame_index, frame, results, loop_epoch)
                if loop_epoch != runtime.loop_epoch:
                    continue
                self._overlay_fall_annotations(annotated_frame, fall_detection_service.get_track_annotations(runtime.camera_id))
                elapsed_ms = (perf_counter() - start) * 1000.0

                runtime.pose_stats.record(elapsed_ms)
                runtime.pose_stats.latest_keypoints = keypoints

                with runtime.frame_lock:
                    runtime.latest_annotated_frame = annotated_frame
                    runtime.latest_frame_jpeg = self._encode_frame(annotated_frame)

                self._set_state(
                    runtime.camera_id,
                    status='looping' if runtime.loop_on_eof else 'live',
                    pose_process_fps=runtime.pose_stats.process_fps,
                    pose_last_ms=runtime.pose_stats.last_ms,
                    pose_error=None,
                    latest_keypoints=runtime.pose_stats.latest_keypoints,
                )

                target_interval = 1.0 / max(1.0, runtime.target_process_fps)
                elapsed_seconds = elapsed_ms / 1000.0
                remaining = target_interval - elapsed_seconds
                if remaining > 0 and runtime.stop_event.wait(remaining):
                    break
            except Exception as exc:
                runtime.pose_stats.error = f'inference failed: {exc}'
                self._set_state(runtime.camera_id, status='error', pose_error=runtime.pose_stats.error)
                runtime.stop_event.set()
                self._signal_end_of_stream(runtime)
                break

    def _set_state(self, camera_id: str, **updates: Any) -> None:
        with state.lock:
            frame_source = state.frame_sources.get(camera_id)
            if frame_source is not None:
                frame_source.update(updates)

    def _camera_name(self, camera_id: str) -> str:
        with state.lock:
            camera = state.cameras.get(camera_id)
            if camera is None:
                return camera_id

            return str(camera.get('name') or camera_id)

    def _reset_loop_runtime(self, runtime: CameraRuntime) -> None:
        with runtime.frame_lock:
            runtime.latest_frame = None
            runtime.latest_annotated_frame = None
            runtime.latest_frame_jpeg = None

        runtime.pose_stats = PoseStats()

        while True:
            try:
                runtime.pose_queue.get_nowait()
            except Empty:
                break

    def _overlay_fall_annotations(self, frame: np.ndarray, annotations: list[dict[str, Any]]) -> None:
        height, width = frame.shape[:2]

        for annotation in annotations:
            box = annotation.get('box')
            if not box or len(box) != 4:
                continue

            x1, y1, x2, y2 = [int(round(value)) for value in box]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            is_fall = bool(annotation.get('fall_active', False))
            color = (0, 0, 255) if is_fall else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    def _signal_end_of_stream(self, runtime: CameraRuntime) -> None:
        try:
            runtime.pose_queue.put_nowait(None)
            return
        except Full:
            pass

        try:
            runtime.pose_queue.get_nowait()
        except Empty:
            return

        try:
            runtime.pose_queue.put_nowait(None)
        except Full:
            pass

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if success:
            return encoded.tobytes()

        return self._placeholder_bytes()

    def _placeholder_bytes(self) -> bytes:
        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        canvas[:] = (17, 24, 39)
        cv2.rectangle(canvas, (24, 24), (616, 336), (51, 65, 85), 3)
        cv2.putText(canvas, 'Processing...', (180, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (248, 250, 252), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Frame source warming up', (145, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (148, 163, 184), 2, cv2.LINE_AA)
        success, encoded = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return encoded.tobytes() if success else b''

    def _coerce_capture_source(self, source: str, source_kind: str) -> Any:
        if source_kind == 'camera' and source.isdigit():
            return int(source)
        return source