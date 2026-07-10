import { useEffect, useMemo, useRef, useState } from 'react'
import { Navigate, Route, Routes, useNavigate } from 'react-router-dom'
import IncidentReportsPage from './pages/IncidentReportsPage'

type Camera = {
  id: string
  name: string
  status: string
  isLive: boolean
  stream_name?: string
}

type AlertEvent = {
  alert_id?: string
  camera_id: string
  camera_name: string
  confidence: number
  timestamp: number
}

const localCameras: Camera[] = [
  {
    id: 'main_webcam_0',
    name: 'Main Webcam',
    status: 'Monitoring',
    isLive: true,
    stream_name: 'Main Hall Stream',
  },
  {
    id: 'hallway_cam_1',
    name: 'Hallway Camera',
    status: 'Offline',
    isLive: false,
    stream_name: 'Hallway Stream',
  },
]

const getServerBaseUrl = () => {
  const configuredBaseUrl = import.meta.env.VITE_SERVER_BASE_URL ?? import.meta.env.VITE_API_BASE_URL
  if (typeof configuredBaseUrl === 'string' && configuredBaseUrl.trim()) {
    return configuredBaseUrl.replace(/\/$/, '')
  }

  if (typeof window !== 'undefined') {
    const host = window.location.hostname
    if (host && host !== 'localhost') {
      return `http://${host}:8000`
    }
  }

  return 'http://localhost:8000'
}

function getStatusClass(camera: Camera) {
  const normalized = camera.status.toLowerCase()
  if (normalized.includes('offline')) return 'status-offline'
  if (normalized.includes('fall')) return 'status-fall'
  return camera.isLive ? 'status-live' : 'status-offline'
}

function HomePage() {
  const navigate = useNavigate()
  const [cameras, setCameras] = useState<Camera[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showCameraStatus, setShowCameraStatus] = useState(false)
  const [lastAlert, setLastAlert] = useState<AlertEvent | null>(null)
  const [notificationState, setNotificationState] = useState<NotificationPermission | 'unsupported'>(() => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      return 'unsupported'
    }
    return Notification.permission
  })

  const alertIdsRef = useRef<Set<string>>(new Set())

  const loadCameras = async () => {
    try {
      const response = await fetch(`${getServerBaseUrl()}/api/cameras`, {
        method: 'GET',
        cache: 'no-store',
      })
      if (!response.ok) {
        throw new Error(`camera list failed with status ${response.status}`)
      }

      const data = await response.json()
      setCameras(Array.isArray(data.cameras) ? data.cameras : [])
    } catch {
      setCameras(localCameras)
    } finally {
      setIsLoading(false)
    }
  }

  const notifyFall = (alert: AlertEvent) => {
    if (notificationState !== 'granted') return

    try {
      const notification = new Notification('Fall Detected', {
        body: `${alert.camera_name} - Confidence ${(alert.confidence * 100).toFixed(1)}%`,
        tag: `fall-${alert.camera_id}`,
      })

      notification.onclick = () => {
        window.focus()
        setShowCameraStatus(true)
      }
    } catch {
      // Notification can fail on restricted browsers.
    }
  }

  const playAlertSound = () => {
    try {
      const webkitAudioContext = (
        window as typeof window & { webkitAudioContext?: typeof AudioContext }
      ).webkitAudioContext
      const AudioContextCtor = window.AudioContext || webkitAudioContext
      if (!AudioContextCtor) return

      const audioContext = new AudioContextCtor()
      const oscillator = audioContext.createOscillator()
      const gainNode = audioContext.createGain()

      oscillator.connect(gainNode)
      gainNode.connect(audioContext.destination)

      oscillator.frequency.setValueAtTime(840, audioContext.currentTime)
      oscillator.frequency.setValueAtTime(620, audioContext.currentTime + 0.12)
      oscillator.frequency.setValueAtTime(840, audioContext.currentTime + 0.24)

      gainNode.gain.setValueAtTime(0.24, audioContext.currentTime)
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5)

      oscillator.start(audioContext.currentTime)
      oscillator.stop(audioContext.currentTime + 0.5)
    } catch {
      // Ignore browser audio errors.
    }
  }

  const consumeAlert = (alert: AlertEvent) => {
    const alertKey = alert.alert_id || `${alert.camera_id}_${Math.floor(alert.timestamp / 10)}`
    if (alertIdsRef.current.has(alertKey)) {
      return
    }

    alertIdsRef.current.add(alertKey)
    setLastAlert(alert)
    setShowCameraStatus(true)
    playAlertSound()
    notifyFall(alert)

    window.setTimeout(() => {
      alertIdsRef.current.delete(alertKey)
    }, 30000)
  }

  const requestNotificationPermission = async () => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      setNotificationState('unsupported')
      return
    }

    const result = await Notification.requestPermission()
    setNotificationState(result)
  }

  useEffect(() => {
    loadCameras()
    const timer = window.setInterval(loadCameras, 5000)

    return () => {
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    let source: EventSource | null = null

    try {
      source = new EventSource(`${getServerBaseUrl()}/api/alerts/stream`)
      source.addEventListener('alert', (event) => {
        try {
          const alert = JSON.parse((event as MessageEvent).data) as AlertEvent
          consumeAlert(alert)
        } catch {
          // Ignore malformed SSE payloads.
        }
      })
    } catch {
      source = null
    }

    return () => {
      if (source) {
        source.close()
      }
    }
  }, [notificationState])

  const openIncidentReports = () => {
    const popup = window.open('/incidents', 'fallguard-incidents', 'popup=yes,width=440,height=760')
    if (!popup) {
      navigate('/incidents')
      return
    }

    popup.focus()
  }

  const activeCount = useMemo(() => cameras.filter((camera) => camera.isLive).length, [cameras])
  const lastAlertStreamName = useMemo(() => {
    if (!lastAlert) return null

    const matchedCamera = cameras.find((camera) => camera.id === lastAlert.camera_id)
    return matchedCamera?.stream_name || null
  }, [cameras, lastAlert])

  return (
    <div className="page-shell">
      <div className="content-wrap">
        <header className="panel glass">
          <div className="brand-row">
            <div className="brand-dot" aria-hidden="true">
              <img src="/static/images/fallguard-logo.png" alt="" className="brand-logo" />
            </div>
            <div>
              <h1 className="title">FallGuard Mobile</h1>
              <p className="subtitle">User view for camera status and incidents</p>
            </div>
          </div>
          <div className="header-meta">
            <span className="meta-pill">Cameras Online: {activeCount}</span>
            {notificationState !== 'denied' ? (
              <button
                className="notify-btn"
                onClick={requestNotificationPermission}
                disabled={notificationState === 'granted' || notificationState === 'unsupported'}
              >
                {notificationState === 'granted' && 'Notifications Enabled'}
                {notificationState === 'default' && 'Enable Fall Notifications'}
                {notificationState === 'unsupported' && 'Notifications Unsupported'}
              </button>
            ) : null}
          </div>
        </header>

        {lastAlert ? (
          <section className="panel fall-alert">
            <div className="fall-alert-head">
              <strong>Fall detected on {lastAlert.camera_name}</strong>
              <button
                type="button"
                className="fall-alert-close"
                onClick={() => setLastAlert(null)}
                aria-label="Close fall alert"
              >
                X
              </button>
            </div>
            <span>{lastAlertStreamName || '-'}</span>
            <span>Confidence {(lastAlert.confidence * 100).toFixed(1)}%</span>
          </section>
        ) : null}

        <section className="action-grid">
          <button className="action-card" onClick={() => setShowCameraStatus((prev) => !prev)}>
            <h2>🧿 Camera Status</h2>
            <p>View stream name, camera name, and current status</p>
          </button>

          <button className="action-card" onClick={openIncidentReports}>
            <h2>📋 Incident Reports</h2>
            <p>Open incident history in a separate window</p>
          </button>
        </section>

        {showCameraStatus ? (
          <section className="panel glass">
            <div className="section-header">
              <h3>Camera Status List</h3>
              <button className="link-btn" onClick={loadCameras}>Refresh</button>
            </div>

            {isLoading ? <p className="muted">Loading camera status...</p> : null}
            {!isLoading && cameras.length === 0 ? <p className="muted">No cameras available.</p> : null}

            {cameras.map((camera) => (
              <article className="camera-card" key={camera.id}>
                <p><strong>Stream:</strong> {camera.stream_name || '-'}</p>
                <p><strong>Camera:</strong> {camera.name}</p>
                <p>
                  <strong>Status:</strong>{' '}
                  <span className={`status-pill ${getStatusClass(camera)}`}>{camera.status}</span>
                </p>
              </article>
            ))}
          </section>
        ) : null}
      </div>
    </div>
  )
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/incidents" element={<IncidentReportsPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
