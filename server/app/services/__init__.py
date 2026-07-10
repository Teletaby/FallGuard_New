from app.services.alerts import AlertService
from app.services.auth import AuthService
from app.services.cameras import CameraService
from app.services.fall_detection import FallDetectionService
from app.services.incidents import IncidentService
from app.services.pose import PoseService, pose_service
from app.services.storage import StorageService

alert_service = AlertService()
auth_service = AuthService()
camera_service = CameraService()
incident_service = IncidentService()
fall_detection_service = FallDetectionService(alert_service=alert_service, incident_service=incident_service)
from app.services.frame_loops import FrameLoopService
frame_loop_service = FrameLoopService()
storage_service = StorageService()