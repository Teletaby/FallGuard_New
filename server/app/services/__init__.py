from app.services.alerts import AlertService
from app.services.auth import AuthService
from app.services.cameras import CameraService
from app.services.frame_loops import FrameLoopService
from app.services.incidents import IncidentService
from app.services.pose import PoseService, pose_service
from app.services.storage import StorageService

alert_service = AlertService()
auth_service = AuthService()
camera_service = CameraService()
frame_loop_service = FrameLoopService()
incident_service = IncidentService()
storage_service = StorageService()