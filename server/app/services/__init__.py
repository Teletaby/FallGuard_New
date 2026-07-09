from app.services.alerts import AlertService
from app.services.auth import AuthService
from app.services.cameras import CameraService
from app.services.incidents import IncidentService
from app.services.pose import PoseService
from app.services.storage import StorageService

alert_service = AlertService()
auth_service = AuthService()
camera_service = CameraService()
incident_service = IncidentService()
pose_service = PoseService()
storage_service = StorageService()