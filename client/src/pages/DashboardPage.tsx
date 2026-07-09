import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

type Camera = {
  id: string
  name: string
  status: string
  color: string
  isLive: boolean
  confidence_score: number
  fps?: number
  source?: string
}

type Settings = {
  fall_threshold: number
  fall_delay_seconds: number
  privacy_mode: string
  pre_fall_buffer_seconds: number
  hide_overlays?: boolean
}

type TelegramSubscriber = {
  chat_id: string
  name?: string
  username?: string
}

type Incident = {
  id: string
  timestamp: string
  severity: 'HIGH' | 'MEDIUM' | 'LOW'
  location: string
  confidence: number
  notes?: string
}

type Toast = {
  id: string
  message: string
  type: 'success' | 'error' | 'warning' | 'info'
}

const defaultSettings: Settings = {
  fall_threshold: 0.7,
  fall_delay_seconds: 2,
  privacy_mode: 'full_video',
  pre_fall_buffer_seconds: 5,
  hide_overlays: true
}

const ADMIN_AUTH_KEY = 'fallguard_admin_authenticated'
const ADMIN_PASSWORD = 'admin'

const dashboardThemeStyles = `
  .dashboard-theme-light {
    background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    color: #0f172a;
  }
  .dashboard-theme-light .bg-gray-900,
  .dashboard-theme-light .bg-gray-800 {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    border: 1px solid transparent !important;
    box-shadow: inset 0 0 0 1px #dbe4ee !important;
  }
  .dashboard-theme-light .bg-gray-700,
  .dashboard-theme-light .bg-gray-600 {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
  }
  .dashboard-theme-light .bg-gray-800\/40 {
    background-color: rgba(255, 255, 255, 0.75) !important;
  }
  .dashboard-theme-light .text-white {
    color: #0f172a !important;
  }
  .dashboard-theme-light .text-gray-300 {
    color: #334155 !important;
  }
  .dashboard-theme-light .text-gray-400 {
    color: #64748b !important;
  }
  .dashboard-theme-light .text-gray-500 {
    color: #94a3b8 !important;
  }
  .dashboard-theme-light .border-gray-700,
  .dashboard-theme-light .border-gray-600 {
    border-color: #dbe4ee !important;
  }
  .dashboard-theme-light .hover\:bg-gray-700:hover,
  .dashboard-theme-light .hover\:bg-gray-600:hover {
    background-color: #e2e8f0 !important;
  }
  .dashboard-theme-light .hover\:text-white:hover {
    color: #0f172a !important;
  }
  .dashboard-theme-light .theme-toggle-button {
    background-color: #e2e8f0 !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
  }
  .dashboard-theme-light .theme-toggle-button:hover {
    background-color: #f8fafc !important;
  }
  .dashboard-theme-light .bg-blue-600,
  .dashboard-theme-light .bg-green-600,
  .dashboard-theme-light .bg-red-600,
  .dashboard-theme-light .bg-yellow-600,
  .dashboard-theme-light .bg-purple-600 {
    color: #ffffff !important;
  }
  .dashboard-theme-light .bg-black\/70 {
    background-color: rgba(15, 23, 42, 0.2) !important;
  }
  .dashboard-theme-light .overflow-y-auto,
  .dashboard-theme-light .overflow-y-scroll,
  .dashboard-theme-light .overflow-auto,
  .dashboard-theme-light .overflow-scroll {
    scrollbar-gutter: stable both-edges;
  }
  .dashboard-theme-night .overflow-y-auto,
  .dashboard-theme-night .overflow-y-scroll,
  .dashboard-theme-night .overflow-auto,
  .dashboard-theme-night .overflow-scroll {
    scrollbar-gutter: stable both-edges;
  }
  .dashboard-theme-night .bg-gray-900,
  .dashboard-theme-night .bg-gray-800 {
    background-color: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid transparent !important;
    box-shadow: inset 0 0 0 1px rgba(51, 65, 85, 0.95) !important;
  }
  .dashboard-theme-night .bg-gray-800.rounded-xl,
  .dashboard-theme-night .bg-gray-800.rounded-2xl,
  .dashboard-theme-night .bg-gray-800.rounded-lg,
  .dashboard-theme-night .bg-gray-900.rounded-xl,
  .dashboard-theme-night .bg-gray-900.rounded-2xl,
  .dashboard-theme-night .bg-gray-900.rounded-lg,
  .dashboard-theme-night .bg-gray-700.rounded-xl,
  .dashboard-theme-night .bg-gray-700.rounded-2xl,
  .dashboard-theme-night .bg-gray-700.rounded-lg,
  .dashboard-theme-night .bg-gray-600.rounded-xl,
  .dashboard-theme-night .bg-gray-600.rounded-2xl,
  .dashboard-theme-night .bg-gray-600.rounded-lg {
    border: 0.8px solid rgba(71, 85, 105, 0.9) !important;
    box-shadow: inset 0 0 0 1px rgba(71, 85, 105, 0.6) !important;
  }
  .dashboard-theme-night .bg-gray-700,
  .dashboard-theme-night .bg-gray-600 {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid transparent !important;
    box-shadow: inset 0 0 0 1px rgba(71, 85, 105, 0.9) !important;
  }
  .dashboard-theme-night .bg-gray-800\/40 {
    background-color: rgba(15, 23, 42, 0.85) !important;
    box-shadow: inset 0 0 0 1px rgba(71, 85, 105, 0.55) !important;
  }
  .dashboard-theme-night .theme-toggle-button {
    background-color: #334155 !important;
    color: #f8fafc !important;
    border-color: #64748b !important;
  }
  .dashboard-theme-night .theme-toggle-button:hover {
    background-color: #475569 !important;
    border-color: #94a3b8 !important;
  }
  .dashboard-theme-night .rounded-xl,
  .dashboard-theme-night .rounded-2xl,
  .dashboard-theme-night .rounded-lg {
    box-shadow: inset 0 0 0 1px rgba(71, 85, 105, 0.9), 0 12px 30px rgba(2, 6, 23, 0.35) !important;
  }
  .dashboard-theme-night .shadow-2xl,
  .dashboard-theme-night .shadow-xl,
  .dashboard-theme-night .shadow-lg {
    box-shadow: inset 0 0 0 1px rgba(71, 85, 105, 0.9), 0 20px 45px rgba(2, 6, 23, 0.55) !important;
  }
  .dashboard-theme-light .rounded-xl,
  .dashboard-theme-light .rounded-2xl,
  .dashboard-theme-light .rounded-lg {
    box-shadow: inset 0 0 0 1px #dbe4ee, 0 12px 30px rgba(15, 23, 42, 0.06) !important;
  }
  .dashboard-theme-light .shadow-2xl,
  .dashboard-theme-light .shadow-xl,
  .dashboard-theme-light .shadow-lg {
    box-shadow: inset 0 0 0 1px #dbe4ee, 0 20px 45px rgba(15, 23, 42, 0.1) !important;
  }
`

const buildThemeToggleLabel = (theme: 'light' | 'night') => (theme === 'light' ? 'Night Mode' : 'Light Mode')

const localBackend = {
  settings: { ...defaultSettings },
  telegramToken: '',
  telegramBotName: 'FallGuard Local Bot',
  cameras: [
    {
      id: 'main_webcam_0',
      name: 'Main Webcam',
      status: 'Active',
      color: 'green',
      isLive: true,
      confidence_score: 0.82,
      fps: 29.8,
      source: 'Local webcam 0'
    },
    {
      id: 'hallway_cam_1',
      name: 'Hallway Camera',
      status: 'Monitoring',
      color: 'green',
      isLive: true,
      confidence_score: 0.41,
      fps: 24.3,
      source: 'rtsp://local/hallway'
    }
  ] as Camera[],
  subscribers: [
    { chat_id: '10001', name: 'Local Admin', username: 'admin' },
    { chat_id: '10002', name: 'Nurse Station', username: 'nurse_station' }
  ] as TelegramSubscriber[],
  blocked: ['10003'],
  incidents: [
    {
      id: 'INC-1001',
      timestamp: '2026-07-08 09:15:00',
      severity: 'HIGH',
      location: 'Main Hall',
      confidence: 0.94,
      notes: 'Local sample incident for offline mode.'
    },
    {
      id: 'INC-1002',
      timestamp: '2026-07-08 10:42:00',
      severity: 'MEDIUM',
      location: 'Stairwell',
      confidence: 0.73
    }
  ] as Incident[]
}

const cloneCamera = (camera: Camera): Camera => ({ ...camera })
const cloneSubscriber = (subscriber: TelegramSubscriber): TelegramSubscriber => ({ ...subscriber })
const cloneIncident = (incident: Incident): Incident => ({ ...incident })

const buildPlaceholderFeed = (title: string, subtitle: string) => {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#111827" />
          <stop offset="100%" stop-color="#1f2937" />
        </linearGradient>
      </defs>
      <rect width="1280" height="720" fill="url(#g)" />
      <rect x="72" y="72" width="1136" height="576" rx="28" fill="#0f172a" stroke="#334155" stroke-width="4" />
      <text x="640" y="320" text-anchor="middle" fill="#f8fafc" font-family="Inter, Arial, sans-serif" font-size="64" font-weight="700">${title}</text>
      <text x="640" y="390" text-anchor="middle" fill="#94a3b8" font-family="Inter, Arial, sans-serif" font-size="30">${subtitle}</text>
      <text x="640" y="490" text-anchor="middle" fill="#38bdf8" font-family="Inter, Arial, sans-serif" font-size="22" letter-spacing="3">LOCAL VIEW · NO API</text>
    </svg>
  `

  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

const escapePdfText = (value: string) => value.replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)')

const buildIncidentPdf = (incident: Incident, subscribers: TelegramSubscriber[]) => {
  const lines = [
    'FALL DETECTION INCIDENT REPORT',
    `Incident ID: ${incident.id}`,
    `Timestamp: ${incident.timestamp}`,
    `Severity: ${incident.severity}`,
    `Location: ${incident.location}`,
    `Confidence: ${(incident.confidence * 100).toFixed(1)}%`,
    `Notes: ${incident.notes || 'None'}`,
    '',
    'Telegram recipients:',
    ...(subscribers.length > 0
      ? subscribers.map((subscriber) => `${subscriber.name || 'Unknown'} - ${subscriber.chat_id}`)
      : ['No subscribers configured'])
  ]

  const content = [
    'BT',
    '/F1 18 Tf',
    '50 760 Td',
    `(${escapePdfText(lines[0])}) Tj`
  ]

  let yOffset = 730
  for (const line of lines.slice(1)) {
    if (line === '') {
      yOffset -= 14
      continue
    }

    content.push('/F1 11 Tf')
    content.push(`50 ${yOffset} Td`)
    content.push(`(${escapePdfText(line)}) Tj`)
    yOffset -= 22
  }

  content.push('ET')
  const contentStream = content.join('\n')

  const objects = [
    '1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj',
    '2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj',
    '3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj',
    '4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj',
    `5 0 obj << /Length ${contentStream.length} >> stream\n${contentStream}\nendstream endobj`
  ]

  const header = '%PDF-1.4\n'
  let body = ''
  const offsets = [0]

  for (const object of objects) {
    offsets.push(header.length + body.length)
    body += `${object}\n`
  }

  const xrefStart = header.length + body.length
  const xref = ['xref', '0 6', '0000000000 65535 f ']
  for (let index = 1; index < offsets.length; index += 1) {
    xref.push(`${String(offsets[index]).padStart(10, '0')} 00000 n `)
  }

  const trailer = ['trailer << /Size 6 /Root 1 0 R >>', 'startxref', String(xrefStart), '%%EOF']
  return `${header}${body}${xref.join('\n')}\n${trailer.join('\n')}`
}

function DashboardPage() {
  const navigate = useNavigate()
  const [theme, setTheme] = useState<'light' | 'night'>(() => {
    if (typeof window === 'undefined') return 'light'
    return (window.localStorage.getItem('fallguard_dashboard_theme') as 'light' | 'night' | null) || 'light'
  })
  const [cameras, setCameras] = useState<Camera[]>([])
  const [mainStreamId, setMainStreamId] = useState('main_webcam_0')
  const [mainStreamError, setMainStreamError] = useState(false)
  const [settings, setSettings] = useState<Settings>(defaultSettings)
  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(false)
  const [showAdminLogin, setShowAdminLogin] = useState(false)
  const [showAdminPanel, setShowAdminPanel] = useState(false)
  const [showCameraManager, setShowCameraManager] = useState(false)
  const [showTelegramSubscribers, setShowTelegramSubscribers] = useState(false)
  const [adminPassword, setAdminPassword] = useState('')
  const [telegramToken, setTelegramToken] = useState('')
  const [telegramStatusText, setTelegramStatusText] = useState('')
  const [telegramTestDisabled, setTelegramTestDisabled] = useState(true)
  const [adminSubscribers, setAdminSubscribers] = useState<TelegramSubscriber[]>([])
  const [adminBlocked, setAdminBlocked] = useState<string[]>([])
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [cameraDefinitions, setCameraDefinitions] = useState<Camera[]>([])
  const [toasts, setToasts] = useState<Toast[]>([])
  const [globalAlert, setGlobalAlert] = useState<{
    cameraId: string
    cameraName: string
    confidence: number
  } | null>(null)
  const [uploadProgress, setUploadProgress] = useState({
    percent: 0,
    text: 'Starting upload...'
  })
  const [isUploading, setIsUploading] = useState(false)
  const [cameraName, setCameraName] = useState('')
  const [cameraSource, setCameraSource] = useState('')
  const [uploadName, setUploadName] = useState('')
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [manualChatId, setManualChatId] = useState('')
  const [manualName, setManualName] = useState('')
  const [activeTab, setActiveTab] = useState<'webcam' | 'upload'>('webcam')

  const alertMapRef = useRef<Map<string, { cameraId: string; timestamp: number }>>(new Map())
  const pollingRef = useRef<number | null>(null)
  const alertPollingRef = useRef<number | null>(null)
  const uploadTimerRef = useRef<number | null>(null)

  const mainCamera = useMemo(
    () => cameras.find((cam) => cam.id === mainStreamId),
    [cameras, mainStreamId]
  )

  const statusCounts = useMemo(() => {
    const activeCameras = cameras.filter((cam) => cam.isLive).length
    const fallDetections = cameras.filter((cam) => cam.color === 'red').length
    return { activeCameras, fallDetections }
  }, [cameras])

  useEffect(() => {
    startPolling()
    return () => stopPolling()
  }, [])

  useEffect(() => {
    if (showCameraManager) {
      loadCameraStatusList()
    }
  }, [showCameraManager])

  useEffect(() => {
    if (showTelegramSubscribers) {
      loadTelegramSubscribers()
    }
  }, [showTelegramSubscribers])

  useEffect(() => {
    setIsAdminAuthenticated(window.localStorage.getItem(ADMIN_AUTH_KEY) === 'true')
  }, [])

  useEffect(() => {
    window.localStorage.setItem('fallguard_dashboard_theme', theme)
  }, [theme])

  const apiCall = async (endpoint: string, options: RequestInit = {}) => {
    const method = (options.method || 'GET').toUpperCase()
    const rawBody = options.body
    const body = rawBody instanceof FormData
      ? Object.fromEntries(Array.from(rawBody.entries()))
      : typeof rawBody === 'string'
      ? (() => {
          try {
            return JSON.parse(rawBody)
          } catch {
            return {}
          }
        })()
      : {}

    if (endpoint === '/settings') {
      if (method === 'POST') {
        localBackend.settings = {
          ...localBackend.settings,
          ...(body.fall_threshold !== undefined ? { fall_threshold: Number(body.fall_threshold) } : {}),
          ...(body.fall_delay_seconds !== undefined ? { fall_delay_seconds: Number(body.fall_delay_seconds) } : {}),
          ...(body.privacy_mode !== undefined ? { privacy_mode: String(body.privacy_mode) } : {}),
          ...(body.pre_fall_buffer_seconds !== undefined ? { pre_fall_buffer_seconds: Number(body.pre_fall_buffer_seconds) } : {}),
          ...(body.hide_overlays !== undefined ? { hide_overlays: Boolean(body.hide_overlays) } : {})
        }
        return { success: true, settings: { ...localBackend.settings } }
      }

      return {
        success: true,
        settings: { ...localBackend.settings },
        telegram_token: Boolean(localBackend.telegramToken),
        telegram_bot_name: localBackend.telegramBotName
      }
    }

    if (endpoint === '/alerts/active') {
      return {
        success: true,
        alerts: localBackend.cameras
          .filter((camera) => camera.color === 'red' && camera.isLive)
          .map((camera) => ({
            camera_id: camera.id,
            camera_name: camera.name,
            confidence: camera.confidence_score,
            timestamp: Math.floor(Date.now() / 1000)
          }))
      }
    }

    if (endpoint === '/telegram/set_token' && method === 'POST') {
      localBackend.telegramToken = String(body.token || '').trim()
      return { success: Boolean(localBackend.telegramToken), message: 'Telegram bot token saved', bot_username: 'local_bot' }
    }

    if (endpoint === '/telegram/test_alert' && method === 'POST') {
      if (!localBackend.telegramToken) {
        return { success: false, message: 'Telegram bot not configured' }
      }

      if (localBackend.subscribers.length === 0) {
        return { success: false, message: 'No subscribers' }
      }

      return { success: true, message: 'Test alerts sent', sent_count: localBackend.subscribers.length }
    }

    if (endpoint === '/telegram/subscribers') {
      return { success: true, subscribers: localBackend.subscribers.map(cloneSubscriber) }
    }

    if (endpoint === '/telegram/add_subscriber' && method === 'POST') {
      const chatId = String(body.chat_id || '').trim()
      const name = String(body.name || 'Manual Entry').trim()
      if (!chatId) {
        return { success: false, message: 'Chat ID is required' }
      }

      if (localBackend.subscribers.some((subscriber) => subscriber.chat_id === chatId)) {
        return { success: false, message: 'Subscriber already exists' }
      }

      localBackend.subscribers.push({ chat_id: chatId, name, username: '' })
      return { success: true, message: 'Subscriber added' }
    }

    if (endpoint === '/telegram/remove_subscriber' && method === 'POST') {
      const chatId = String(body.chat_id || '').trim()
      localBackend.subscribers = localBackend.subscribers.filter((subscriber) => subscriber.chat_id !== chatId)
      if (chatId && !localBackend.blocked.includes(chatId)) {
        localBackend.blocked.push(chatId)
      }
      return { success: true, message: 'Subscriber removed' }
    }

    if (endpoint === '/telegram/blocked') {
      return { success: true, blocked: [...localBackend.blocked] }
    }

    if (endpoint === '/telegram/unblock' && method === 'POST') {
      const chatId = String(body.chat_id || '').trim()
      localBackend.blocked = localBackend.blocked.filter((blockedId) => blockedId !== chatId)
      return { success: true, message: 'User unblocked' }
    }

    if (endpoint === '/cameras') {
      return { success: true, cameras: localBackend.cameras.map(cloneCamera) }
    }

    if (endpoint === '/cameras/all_definitions') {
      return { success: true, definitions: localBackend.cameras.map(cloneCamera) }
    }

    if (endpoint === '/cameras/add' && method === 'POST') {
      const name = String(body.name || '').trim()
      const source = String(body.source || '').trim()
      const cameraId = `cam_${Math.random().toString(36).slice(2, 10)}`
      localBackend.cameras.push({
        id: cameraId,
        name,
        status: 'Monitoring',
        color: 'green',
        isLive: true,
        confidence_score: 0.18,
        fps: 24,
        source
      })
      return { success: true, message: `Camera '${name}' added`, camera_id: cameraId }
    }

    if (endpoint.startsWith('/cameras/stop/') && method === 'POST') {
      const cameraId = endpoint.split('/').pop() || ''
      const camera = localBackend.cameras.find((item) => item.id === cameraId)
      if (!camera) {
        return { success: false, message: 'Camera not found' }
      }

      camera.isLive = false
      camera.status = 'Offline'
      camera.color = 'gray'
      camera.fps = 0
      return { success: true, message: 'Camera stopped' }
    }

    if (endpoint.startsWith('/cameras/remove/') && method === 'DELETE') {
      const cameraId = endpoint.split('/').pop() || ''
      localBackend.cameras = localBackend.cameras.filter((camera) => camera.id !== cameraId)
      return { success: true, message: 'Camera removed' }
    }

    if (endpoint === '/cameras/add_existing' && method === 'POST') {
      const cameraId = String(body.camera_id || '').trim()
      const camera = localBackend.cameras.find((item) => item.id === cameraId)
      if (!camera) {
        return { success: false, message: 'Camera not found' }
      }

      camera.isLive = true
      camera.status = 'Monitoring'
      camera.color = camera.color === 'gray' ? 'green' : camera.color
      camera.fps = camera.fps || 24
      return { success: true, message: 'Camera restarted' }
    }

    if (endpoint === '/cameras/upload' && method === 'POST') {
      const name = String(body.get?.('name') || body.name || 'Uploaded Video')
      const cameraId = `cam_${Math.random().toString(36).slice(2, 10)}`
      localBackend.cameras.push({
        id: cameraId,
        name,
        status: 'Processing',
        color: 'green',
        isLive: true,
        confidence_score: 0.22,
        fps: 18,
        source: 'Uploaded video file'
      })
      return { success: true, message: `Video '${name}' uploaded successfully`, camera_id: cameraId }
    }

    if (endpoint === '/incidents') {
      return { success: true, incidents: localBackend.incidents.map(cloneIncident) }
    }

    if (endpoint.startsWith('/incidents/') && endpoint.endsWith('/notes') && method === 'POST') {
      const incidentId = endpoint.split('/')[2]
      const incident = localBackend.incidents.find((item) => item.id === incidentId)
      if (!incident) {
        return { success: false, message: 'Incident not found' }
      }

      incident.notes = String(body.notes || '')
      return { success: true, message: 'Notes updated' }
    }

    if (endpoint.startsWith('/incidents/') && method === 'DELETE') {
      const incidentId = endpoint.split('/')[2]
      localBackend.incidents = localBackend.incidents.filter((incident) => incident.id !== incidentId)
      return { success: true, message: 'Incident deleted' }
    }

    return { success: true }
  }

  const showToast = (message: string, type: Toast['type'] = 'success') => {
    const id = `${Date.now()}-${Math.random()}`
    setToasts((prev) => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id))
    }, 3000)
  }

  const startPolling = () => {
    loadCameras()
    startAlertPolling()

    pollingRef.current = window.setInterval(() => {
      loadCameras()
      if (isAdminAuthenticated && showAdminPanel) {
        loadCameraStatusList()
      }
    }, 2500)
  }

  const stopPolling = () => {
    if (pollingRef.current) {
      window.clearInterval(pollingRef.current)
      pollingRef.current = null
    }
    stopAlertPolling()
  }

  const startAlertPolling = () => {
    checkForWebsiteAlerts()
    alertPollingRef.current = window.setInterval(checkForWebsiteAlerts, 1000)
  }

  const stopAlertPolling = () => {
    if (alertPollingRef.current) {
      window.clearInterval(alertPollingRef.current)
      alertPollingRef.current = null
    }
  }

  const checkForWebsiteAlerts = async () => {
    try {
      const data = await apiCall('/alerts/active')

      if (data.success && Array.isArray(data.alerts)) {
        data.alerts.forEach((alert: { camera_id: string; camera_name: string; confidence: number; timestamp: number }) => {
          const alertKey = `${alert.camera_id}_${Math.floor(alert.timestamp / 10)}`
          if (!alertMapRef.current.has(alertKey)) {
            alertMapRef.current.set(alertKey, {
              cameraId: alert.camera_id,
              timestamp: alert.timestamp
            })
            showGlobalFallAlert(alert.camera_name, alert.confidence, alert.camera_id)
            window.setTimeout(() => alertMapRef.current.delete(alertKey), 30000)
          }
        })

        const now = Date.now() / 1000
        Array.from(alertMapRef.current.entries()).forEach(([key, value]) => {
          if (now - value.timestamp > 35) {
            alertMapRef.current.delete(key)
          }
        })
      }
    } catch {
      // ignore alert errors
    }
  }

  const showGlobalFallAlert = (cameraName: string, confidence: number, cameraId: string) => {
    setGlobalAlert({ cameraName, confidence, cameraId })
    playAlertSound()
    flashPageTitle()
  }

  const dismissGlobalAlert = () => setGlobalAlert(null)

  const switchToFallCamera = () => {
    if (globalAlert && cameras.some((cam) => cam.id === globalAlert.cameraId)) {
      switchMainFeed(globalAlert.cameraId)
      dismissGlobalAlert()
      showToast(`Switched to ${getCameraName(globalAlert.cameraId)}`)
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

      oscillator.frequency.setValueAtTime(800, audioContext.currentTime)
      oscillator.frequency.setValueAtTime(600, audioContext.currentTime + 0.1)
      oscillator.frequency.setValueAtTime(800, audioContext.currentTime + 0.2)

      gainNode.gain.setValueAtTime(0.3, audioContext.currentTime)
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5)

      oscillator.start(audioContext.currentTime)
      oscillator.stop(audioContext.currentTime + 0.5)
    } catch {
      // ignore audio errors
    }
  }

  const flashPageTitle = () => {
    let count = 0
    const originalTitle = document.title
    const flashInterval = window.setInterval(() => {
      document.title = count % 2 === 0 ? '🚨 FALL DETECTED!' : originalTitle
      count += 1
      if (count >= 10) {
        document.title = originalTitle
        window.clearInterval(flashInterval)
      }
    }, 500)
  }

  const getCameraName = (cameraId: string) => {
    const camera = cameras.find((cam) => cam.id === cameraId)
    return camera ? camera.name : 'Unknown Camera'
  }

  const loadCameras = async () => {
    try {
      const data = await apiCall('/cameras')
      const newCameras: Camera[] = data.cameras || []
      setCameras(newCameras)

      if (!newCameras.find((cam) => cam.id === mainStreamId)) {
        const mainCamera = newCameras.find((cam) => cam.id === 'main_webcam_0')
        setMainStreamId(mainCamera ? mainCamera.id : newCameras[0]?.id || 'main_webcam_0')
      }
    } catch (error) {
      console.error('[CAMERAS] Failed to load cameras:', error)
    }
  }

  const switchMainFeed = (cameraId: string) => {
    const camera = cameras.find((cam) => cam.id === cameraId)
    if (!camera) {
      showToast('Camera not found or is offline', 'error')
      return
    }
    if (!camera.isLive) {
      showToast(`Camera "${camera.name}" is currently offline`, 'warning')
      return
    }
    setMainStreamError(false)
    setMainStreamId(cameraId)
  }

  const checkAdminAuth = async () => {
    const authenticated = window.localStorage.getItem(ADMIN_AUTH_KEY) === 'true'
    setIsAdminAuthenticated(authenticated)
    return authenticated
  }

  const openAdminPanel = async () => {
    const isAuth = await checkAdminAuth()
    if (isAuth) {
      setShowAdminPanel(true)
      await Promise.all([
        loadSettings(),
        loadTelegramStatus(),
        loadAdminSubscribersList(),
        loadAdminBlockedList(),
        loadCameraStatusList(),
        loadPrivacySettings(),
        loadIncidents()
      ])
    } else {
      setShowAdminLogin(true)
    }
  }

  const closeAdminPanel = () => setShowAdminPanel(false)

  const logoutAdmin = async () => {
    window.localStorage.removeItem(ADMIN_AUTH_KEY)
    setIsAdminAuthenticated(false)
    setShowAdminPanel(false)
    showToast('Logged out successfully', 'info')
  }

  const submitAdminLogin = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    try {
      if (adminPassword !== ADMIN_PASSWORD) {
        showToast('Invalid password', 'error')
        setAdminPassword('')
        return
      }

      window.localStorage.setItem(ADMIN_AUTH_KEY, 'true')
      setIsAdminAuthenticated(true)
      setShowAdminLogin(false)
      setAdminPassword('')
      await openAdminPanel()
      showToast('Login successful', 'success')
    } catch {
      showToast('Login failed', 'error')
    }
  }

  const loadSettings = async () => {
    try {
      const data = await apiCall('/settings')
      if (data.settings) {
        setSettings((prev) => ({ ...prev, ...data.settings }))
      }
    } catch (error) {
      console.error('Failed to load settings:', error)
    }
  }

  const saveSettings = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    try {
      await apiCall('/settings', {
        method: 'POST',
        body: JSON.stringify({
          fall_threshold: settings.fall_threshold,
          fall_delay_seconds: settings.fall_delay_seconds,
          hide_overlays: !!settings.hide_overlays
        })
      })
      showToast('Settings saved successfully')
      loadCameraStatusList()
    } catch {
      // handled in apiCall
    }
  }

  const loadPrivacySettings = async () => {
    try {
      const data = await apiCall('/settings')
      if (data.settings) {
        setSettings((prev) => ({
          ...prev,
          privacy_mode: data.settings.privacy_mode || prev.privacy_mode,
          pre_fall_buffer_seconds: data.settings.pre_fall_buffer_seconds || prev.pre_fall_buffer_seconds
        }))
      }
    } catch (error) {
      console.error('Failed to load privacy settings:', error)
    }
  }

  const savePrivacySettings = async () => {
    try {
      await apiCall('/settings', {
        method: 'POST',
        body: JSON.stringify({
          privacy_mode: settings.privacy_mode,
          pre_fall_buffer_seconds: settings.pre_fall_buffer_seconds
        })
      })
      showToast('Privacy settings saved successfully')
    } catch {
      // handled
    }
  }

  const loadTelegramStatus = async () => {
    try {
      const data = await apiCall('/settings')
      if (data.telegram_token) {
        setTelegramStatusText(`✅ Connected: ${data.telegram_bot_name || 'Bot'}`)
        setTelegramTestDisabled(false)
      } else {
        setTelegramStatusText('')
        setTelegramTestDisabled(true)
      }
    } catch (error) {
      console.error('Failed to load telegram status:', error)
    }
  }

  const saveTelegramToken = async () => {
    const token = telegramToken.trim()
    if (!token) {
      showToast('Please enter a bot token', 'warning')
      return
    }

    try {
      const response = await apiCall('/telegram/set_token', {
        method: 'POST',
        body: JSON.stringify({ token })
      })

      if (response.success) {
        showToast('Bot token saved successfully!', 'success')
        setTelegramToken('')
        await loadTelegramStatus()
      } else {
        showToast(response.message || 'Failed to save token', 'error')
      }
    } catch {
      showToast('Invalid token or connection failed', 'error')
    }
  }

  const testTelegramAlert = async () => {
    try {
      setTelegramTestDisabled(true)
      const response = await apiCall('/telegram/test_alert', { method: 'POST' })
      if (response.success) {
        showToast(`Test alert sent to ${response.sent_count} subscriber(s)`, 'success')
      } else {
        showToast(response.message || 'Failed to send test alert', 'error')
      }
    } catch {
      showToast('No subscribers configured', 'warning')
    } finally {
      setTelegramTestDisabled(false)
    }
  }

  const loadTelegramSubscribers = async () => {
    try {
      const response = await apiCall('/telegram/subscribers')
      setAdminSubscribers(response.subscribers || [])
    } catch (error) {
      console.error('Failed to load subscribers:', error)
    }
  }

  const loadAdminSubscribersList = async () => loadTelegramSubscribers()

  const loadAdminBlockedList = async () => {
    try {
      const response = await apiCall('/telegram/blocked')
      setAdminBlocked(response.blocked || [])
    } catch (error) {
      console.error('Failed to load blocked list:', error)
    }
  }

  const addManualSubscriber = async () => {
    if (!manualChatId) {
      showToast('Please enter a Chat ID', 'warning')
      return
    }

    if (!/^[0-9]+$/.test(manualChatId)) {
      showToast('Chat ID must contain only numbers', 'error')
      return
    }

    try {
      const response = await apiCall('/telegram/add_subscriber', {
        method: 'POST',
        body: JSON.stringify({
          chat_id: manualChatId,
          name: manualName || 'Manual Entry'
        })
      })

      if (response.success) {
        showToast('Subscriber added successfully!', 'success')
        setManualChatId('')
        setManualName('')
        await loadTelegramSubscribers()
        await loadAdminSubscribersList()
      } else {
        showToast(response.message || 'Failed to add subscriber', 'error')
      }
    } catch {
      showToast('Error adding subscriber', 'error')
    }
  }

  const removeSubscriber = async (chatId: string) => {
    if (!confirm('Remove this subscriber?')) return

    try {
      const response = await apiCall('/telegram/remove_subscriber', {
        method: 'POST',
        body: JSON.stringify({ chat_id: chatId })
      })

      if (response.success) {
        showToast('Subscriber removed', 'success')
        await loadTelegramSubscribers()
        await loadAdminSubscribersList()
        await loadAdminBlockedList()
      } else {
        showToast('Failed to remove subscriber', 'error')
      }
    } catch {
      showToast('Error removing subscriber', 'error')
    }
  }

  const unblockUser = async (chatId: string) => {
    if (!confirm(`Unblock user ${chatId}? They will be able to subscribe again.`)) return

    try {
      const response = await apiCall('/telegram/unblock', {
        method: 'POST',
        body: JSON.stringify({ chat_id: chatId })
      })

      if (response.success) {
        showToast('User unblocked successfully', 'success')
        await loadAdminBlockedList()
      } else {
        showToast('Failed to unblock user', 'error')
      }
    } catch {
      showToast('Error unblocking user', 'error')
    }
  }

  const loadCameraStatusList = async () => {
    try {
      const data = await apiCall('/cameras/all_definitions')
      setCameraDefinitions(data.definitions || [])
    } catch (error) {
      console.error('Failed to load camera status:', error)
    }
  }

  const loadIncidents = async () => {
    try {
      const response = await apiCall('/incidents')
      setIncidents(response.incidents || [])
    } catch (error) {
      console.error('Failed to load incidents:', error)
    }
  }

  const generateIncidentPDF = async (incidentId: string) => {
    try {
      const incident = incidents.find((item) => item.id === incidentId)
      if (!incident) {
        showToast('Incident not found', 'error')
        return
      }

      const pdfBlob = new Blob([buildIncidentPdf(incident, adminSubscribers)], { type: 'application/pdf' })
      const url = window.URL.createObjectURL(pdfBlob)
      const link = document.createElement('a')
      link.href = url
      link.download = `incident_${incidentId}.pdf`
      document.body.appendChild(link)
      link.click()
      window.URL.revokeObjectURL(url)
      link.remove()
      showToast('PDF report downloaded')
    } catch {
      showToast('Error generating PDF', 'error')
    }
  }

  const editIncidentNotes = async (incidentId: string) => {
    const notes = prompt('Enter notes for this incident:')
    if (notes === null) return

    try {
      await apiCall(`/incidents/${incidentId}/notes`, {
        method: 'POST',
        body: JSON.stringify({ notes })
      })
      showToast('Notes updated successfully')
      loadIncidents()
    } catch {
      showToast('Failed to update notes', 'error')
    }
  }

  const deleteIncident = async (incidentId: string) => {
    if (!confirm('Delete this incident report? This cannot be undone.')) return

    try {
      await apiCall(`/incidents/${incidentId}`, { method: 'DELETE' })
      showToast('Incident deleted')
      loadIncidents()
    } catch {
      showToast('Failed to delete incident', 'error')
    }
  }

  const addCamera = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    try {
      await apiCall('/cameras/add', {
        method: 'POST',
        body: JSON.stringify({ name: cameraName, source: cameraSource })
      })
      showToast('Camera added successfully')
      setCameraName('')
      setCameraSource('')
      loadCameras()
      loadCameraStatusList()
    } catch {
      // handled
    }
  }

  const stopCamera = async (id: string) => {
    if (!confirm('Stop this camera stream?')) return
    try {
      await apiCall(`/cameras/stop/${id}`, { method: 'POST' })
      showToast('Camera stopped')
      loadCameras()
      loadCameraStatusList()
    } catch {
      // handled
    }
  }

  const startCamera = async (id: string) => {
    try {
      await apiCall('/cameras/add_existing', {
        method: 'POST',
        body: JSON.stringify({ camera_id: id })
      })
      showToast('Camera started')
      loadCameras()
      loadCameraStatusList()
    } catch {
      // handled
    }
  }

  const removeCamera = async (id: string) => {
    if (!confirm('Permanently remove this camera? This cannot be undone.')) return

    try {
      await apiCall(`/cameras/remove/${id}`, { method: 'DELETE' })
      showToast('Camera removed')

      if (mainStreamId === id) {
        const remaining = cameras.filter((cam) => cam.id !== id)
        if (remaining.length > 0) {
          switchMainFeed(remaining[0].id)
        }
      }

      loadCameras()
      loadCameraStatusList()
    } catch {
      showToast('Failed to remove camera', 'error')
    }
  }

  const submitUpload = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!uploadName || !uploadFile) {
      showToast('Please provide a name and select a video file', 'error')
      return
    }

    if (uploadFile.size > 500 * 1024 * 1024) {
      showToast('File too large. Maximum size is 500MB', 'error')
      return
    }

    setIsUploading(true)
    setUploadProgress({ percent: 0, text: 'Starting upload...' })

    const formData = new FormData()
    formData.append('name', uploadName)
    formData.append('video_file', uploadFile)

    if (uploadTimerRef.current) {
      window.clearInterval(uploadTimerRef.current)
    }

    let percentComplete = 0
    uploadTimerRef.current = window.setInterval(() => {
      percentComplete = Math.min(percentComplete + 20, 95)
      setUploadProgress({ percent: percentComplete, text: `Uploading: ${Math.round(percentComplete)}%` })
    }, 120)

    window.setTimeout(async () => {
      try {
        const data = await apiCall('/cameras/upload', { method: 'POST', body: formData })
        setUploadProgress({ percent: 100, text: 'Processing video...' })
        showToast('Video uploaded successfully!', 'success')
        setUploadName('')
        setUploadFile(null)
        loadCameras()
        loadCameraStatusList()
        if (data.camera_id) {
          switchMainFeed(data.camera_id)
        }
      } catch {
        showToast('Upload response error', 'error')
      } finally {
        if (uploadTimerRef.current) {
          window.clearInterval(uploadTimerRef.current)
          uploadTimerRef.current = null
        }
        setIsUploading(false)
      }
    }, 700)
  }

  const cancelUpload = () => {
    if (uploadTimerRef.current) {
      window.clearInterval(uploadTimerRef.current)
      uploadTimerRef.current = null
    }

    setIsUploading(false)
    setUploadProgress({ percent: 0, text: 'Upload cancelled' })
  }

  const goToHomepage = () => navigate('/landing')

  return (
    <div className={`min-h-screen p-4 ${theme === 'light' ? 'dashboard-theme-light bg-slate-100 text-slate-900' : 'dashboard-theme-night bg-gray-900 text-white'}`}>
      <style>{dashboardThemeStyles}</style>
      {globalAlert ? (
        <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 w-full max-w-2xl">
          <div className="fall-alert-banner bg-red-600 text-white rounded-lg shadow-2xl mx-4">
            <div className="px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="text-3xl urgent-pulse">🚨</div>
                <div>
                  <h3 className="text-xl font-bold">FALL DETECTED!</h3>
                  <p className="text-sm opacity-90">Location: {globalAlert.cameraName}</p>
                  <p className="text-xs opacity-80">Confidence: {(globalAlert.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={switchToFallCamera}
                  className="px-4 py-2 bg-white text-red-600 font-semibold rounded-lg hover:bg-gray-100 transition"
                >
                  View Camera
                </button>
                <button
                  onClick={dismissGlobalAlert}
                  className="text-white hover:text-gray-200 text-2xl font-bold px-3"
                >
                  &times;
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <header className="bg-gray-800 rounded-xl p-4 mb-4 flex items-center justify-between shadow-lg">
        <div className="flex items-center gap-4">
          <button onClick={goToHomepage} className="transition" aria-label="Go to homepage">
            <img src="/static/images/fallguard-logo.svg" alt="FallGuard Logo" className="h-10 w-auto" />
          </button>
          <div>
            <h1 className="text-2xl font-bold">FallGuard Dashboard</h1>
            <p className="text-sm text-gray-400">AI Fall Detection System</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => setTheme((prev) => (prev === 'light' ? 'night' : 'light'))}
            className={`theme-toggle-button px-4 py-2 rounded-lg font-semibold transition border ${theme === 'light' ? 'bg-slate-200 text-slate-900 border-slate-300 hover:bg-slate-100' : 'bg-slate-700 text-slate-100 border-slate-500 hover:bg-slate-600 hover:border-slate-400'}`}
          >
            {buildThemeToggleLabel(theme)}
          </button>
          <div
            className={`status-badge ${
              statusCounts.fallDetections > 0
                ? 'bg-red-600 fall-alert'
                : statusCounts.activeCameras > 0
                ? 'bg-green-600'
                : 'bg-gray-600'
            }`}
          >
            <span className="w-2 h-2 bg-white rounded-full live-indicator"></span>
            <span>
              {statusCounts.fallDetections > 0
                ? '⚠️ FALL DETECTED'
                : statusCounts.activeCameras > 0
                ? 'SYSTEM NORMAL'
                : 'NO CAMERAS'}
            </span>
          </div>
          <button
            onClick={openAdminPanel}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            Admin Panel
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4" style={{ height: 'calc(100vh - 120px)' }}>
        <div className="lg:col-span-2 bg-gray-800 rounded-xl overflow-hidden relative shadow-2xl">
          <div className="main-feed">
            <img
              id="main-stream-img"
              src={buildPlaceholderFeed(mainCamera?.name || 'Main Webcam Stream', mainCamera?.status || 'Active')}
              alt="Main Feed"
              style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
              onLoad={() => setMainStreamError(false)}
              onError={() => setMainStreamError(true)}
            />
            {mainStreamError ? (
              <div className="absolute inset-0 bg-gray-900 flex items-center justify-center flex-col">
                <svg className="w-24 h-24 mb-4 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
                <p className="text-gray-400 text-lg">Camera stream unavailable</p>
                <p className="text-gray-500 text-sm mt-2">Click a camera thumbnail to try again</p>
              </div>
            ) : null}
          </div>

          <div className="absolute top-4 left-4 flex gap-2">
            <div className={`status-badge ${mainCamera?.color === 'red' ? 'bg-red-600 fall-alert' : 'bg-green-600'}`}>
              <span className="w-2 h-2 bg-white rounded-full live-indicator"></span>
              <span>{mainCamera?.color === 'red' ? 'FALL DETECTED' : 'NORMAL'}</span>
            </div>
            {mainCamera && mainCamera.confidence_score > 0.3 ? (
              <div className="status-badge bg-blue-600">
                <span>CONFIDENCE: {(mainCamera.confidence_score * 100).toFixed(1)}%</span>
              </div>
            ) : null}
          </div>

          <div className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/80 to-transparent p-6">
            <h2 className="text-2xl font-bold mb-1">{mainCamera?.name || 'Main Webcam Stream'}</h2>
            <p className="text-gray-300">Status: {mainCamera?.status || 'Active'}</p>
          </div>

          {mainCamera?.color === 'red' ? (
            <div className="absolute bottom-0 left-0 right-0 bg-red-600 text-white shadow-2xl">
              <div className="fall-alert px-6 py-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="text-3xl animate-bounce">⚠️</div>
                  <div>
                    <h3 className="text-xl font-bold">FALL DETECTED!</h3>
                    <p className="text-sm opacity-90">{mainCamera.name}</p>
                    <p className="text-xs opacity-80">Confidence: {(mainCamera.confidence_score * 100).toFixed(1)}%</p>
                  </div>
                </div>
                <button onClick={() => showToast('Alert dismissed', 'info')} className="text-white hover:text-gray-200 text-2xl font-bold px-3">
                  &times;
                </button>
              </div>
            </div>
          ) : null}
        </div>

        <div className="bg-gray-800 rounded-xl p-4 overflow-y-scroll shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-lg">All Cameras</h3>
            <button
              onClick={() => setShowCameraManager(true)}
              className="w-8 h-8 bg-blue-600 hover:bg-blue-700 rounded-full flex items-center justify-center transition"
              title="Add Camera"
            >
              <span className="text-xl">+</span>
            </button>
          </div>

          {cameras.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              <p>No cameras available</p>
              <button
                onClick={() => setShowCameraManager(true)}
                className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition"
              >
                Add Your First Camera
              </button>
            </div>
          ) : (
            <div className="video-grid" id="camera-grid">
              {cameras.map((cam) => (
                <div
                  key={cam.id}
                  className={`camera-thumbnail ${cam.id === mainStreamId ? 'active' : ''} ${cam.color === 'red' ? 'fall-detected' : ''}`}
                  onClick={() => switchMainFeed(cam.id)}
                >
                  {cam.isLive ? (
                    <img src={buildPlaceholderFeed(cam.name, cam.status || 'Live')} alt={cam.name} />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-gray-800 text-gray-400">
                      <p>📷 Camera Offline</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {showAdminLogin ? (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-2xl shadow-2xl w-full max-w-md p-8">
            <h2 className="text-2xl font-bold mb-6 text-center">Admin Login</h2>
            <form onSubmit={submitAdminLogin} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Password</label>
                <input
                  type="password"
                  className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 outline-none"
                  value={adminPassword}
                  onChange={(event) => setAdminPassword(event.target.value)}
                  required
                  autoFocus
                />
              </div>
              <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-lg transition">
                Login
              </button>
              <button
                type="button"
                onClick={() => setShowAdminLogin(false)}
                className="w-full bg-gray-700 hover:bg-gray-600 text-white font-semibold py-3 rounded-lg transition"
              >
                Cancel
              </button>
            </form>
          </div>
        </div>
      ) : null}

      {showAdminPanel ? (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-700 flex items-center justify-between sticky top-0 bg-gray-800 z-10">
              <h2 className="text-2xl font-bold">Admin Panel</h2>
              <div className="flex items-center gap-3">
                <button
                  onClick={logoutAdmin}
                  className="text-sm px-3 py-1 bg-red-600 hover:bg-red-700 rounded-lg font-semibold transition"
                >
                  Logout
                </button>
                <button
                  onClick={closeAdminPanel}
                  className="text-gray-400 hover:text-white text-3xl leading-none"
                >
                  &times;
                </button>
              </div>
            </div>

            <div className="p-6 space-y-6">
              <div className="bg-gray-900 rounded-xl p-6">
                <h3 className="text-lg font-bold mb-4">Detection Parameters</h3>
                <form onSubmit={saveSettings} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Fall Probability Threshold</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                      value={settings.fall_threshold}
                      onChange={(event) =>
                        setSettings((prev) => ({ ...prev, fall_threshold: Number(event.target.value) }))
                      }
                    />
                    <p className="text-xs text-gray-500 mt-1">Confidence level required (0.0 - 1.0)</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Confirmation Delay (seconds)</label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                      value={settings.fall_delay_seconds}
                      onChange={(event) =>
                        setSettings((prev) => ({ ...prev, fall_delay_seconds: Number(event.target.value) }))
                      }
                    />
                    <p className="text-xs text-gray-500 mt-1">Continuous detection time before alert</p>
                  </div>

                  <div className="flex items-start gap-3 bg-gray-800/40 border border-gray-700 rounded-lg px-4 py-3">
                    <input
                      type="checkbox"
                      className="mt-1 h-4 w-4 accent-blue-600"
                      checked={!!settings.hide_overlays}
                      onChange={(event) =>
                        setSettings((prev) => ({ ...prev, hide_overlays: event.target.checked }))
                      }
                    />
                    <div className="flex-1">
                      <label className="block text-sm font-medium">Hide Bounding Box & Skeleton Overlay</label>
                      <p className="text-xs text-gray-500 mt-1">Shows raw video (no on-frame drawings)</p>
                    </div>
                  </div>

                  <button type="submit" className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 rounded-lg transition">
                    Save Settings
                  </button>
                </form>
              </div>

              <div className="bg-gray-900 rounded-xl p-6">
                <h3 className="text-lg font-bold mb-4">🛡️ Privacy & Data Protection</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Privacy Mode</label>
                    <select
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                      value={settings.privacy_mode}
                      onChange={(event) => setSettings((prev) => ({ ...prev, privacy_mode: event.target.value }))}
                    >
                      <option value="full_video">Full Video</option>
                      <option value="skeleton_only">Skeletal Body Only</option>
                      <option value="blurred">Blurred Person View</option>
                      <option value="alerts_only">No Video, Only Alerts</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">Great for care homes and GDPR compliance</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Pre-Fall Video Buffer (seconds)</label>
                    <input
                      type="number"
                      min="1"
                      max="30"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                      value={settings.pre_fall_buffer_seconds}
                      onChange={(event) =>
                        setSettings((prev) => ({ ...prev, pre_fall_buffer_seconds: Number(event.target.value) }))
                      }
                    />
                    <p className="text-xs text-gray-500 mt-1">Video length sent with Telegram alerts (1-30 seconds)</p>
                  </div>

                  <button
                    type="button"
                    onClick={savePrivacySettings}
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 rounded-lg transition"
                  >
                    Save Privacy Settings
                  </button>
                </div>
              </div>

              <div className="bg-gray-900 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold">Telegram Notifications</h3>
                  <button
                    type="button"
                    onClick={() => {
                      setShowTelegramSubscribers(true)
                      loadTelegramSubscribers()
                    }}
                    className="text-3xl hover:text-purple-400 transition"
                    title="Manage Subscribers"
                  >
                    🔔
                  </button>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Bot Token</label>
                    <div className="flex gap-2">
                      <input
                        type="password"
                        placeholder="Paste your Telegram bot token here"
                        className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                        value={telegramToken}
                        onChange={(event) => setTelegramToken(event.target.value)}
                      />
                      <button
                        type="button"
                        onClick={saveTelegramToken}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition"
                      >
                        Save Token
                      </button>
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      Get token from{' '}
                      <a href="https://t.me/BotFather" target="_blank" rel="noreferrer" className="text-blue-400 hover:text-blue-300">
                        /start @BotFather
                      </a>
                      {' '}on Telegram
                    </p>
                  </div>

                  {telegramStatusText ? (
                    <div className="text-sm text-gray-400">
                      <p>{telegramStatusText}</p>
                    </div>
                  ) : null}

                  <button
                    type="button"
                    onClick={testTelegramAlert}
                    disabled={telegramTestDisabled}
                    className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Test Alert
                  </button>
                </div>
              </div>

              <div className="bg-gray-900 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold">📱 Connected Subscribers</h3>
                  <span className="bg-blue-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
                    {adminSubscribers.length}
                  </span>
                </div>

                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {adminSubscribers.length === 0 ? (
                    <p className="text-gray-500 text-sm text-center py-4">No subscribers yet. Users can send /start to your bot!</p>
                  ) : (
                    adminSubscribers.map((sub) => (
                      <div key={sub.chat_id} className="bg-gray-800 p-4 rounded-lg border border-gray-700 hover:border-blue-500 transition">
                        <div className="flex items-center justify-between mb-2">
                          <p className="font-semibold text-white">{sub.name || 'Unknown'}</p>
                          <div className="flex gap-2">
                            <span className="text-xs bg-green-600 text-white px-2 py-1 rounded">✅ Active</span>
                            <button
                              type="button"
                              onClick={() => removeSubscriber(sub.chat_id)}
                              className="text-xs bg-red-600 hover:bg-red-700 text-white px-2 py-1 rounded transition"
                            >
                              Remove
                            </button>
                          </div>
                        </div>
                        <p className="text-xs text-gray-400">Chat ID: {sub.chat_id}</p>
                        {sub.username ? <p className="text-xs text-gray-400">Telegram: @{sub.username}</p> : null}
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="bg-gray-900 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold">🚫 Blocked Users</h3>
                  <span className="bg-red-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
                    {adminBlocked.length}
                  </span>
                </div>

                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {adminBlocked.length === 0 ? (
                    <p className="text-gray-500 text-sm text-center py-4">No blocked users. Removed subscribers will appear here.</p>
                  ) : (
                    adminBlocked.map((chatId) => (
                      <div key={chatId} className="bg-gray-800 p-4 rounded-lg border border-gray-700 hover:border-red-500 transition">
                        <div className="flex items-center justify-between mb-2">
                          <p className="font-semibold text-white">Chat ID: {chatId}</p>
                          <button
                            type="button"
                            onClick={() => unblockUser(chatId)}
                            className="text-xs bg-green-600 hover:bg-green-700 text-white px-2 py-1 rounded transition"
                          >
                            Unblock
                          </button>
                        </div>
                        <p className="text-xs text-gray-400">Blocked from receiving alerts</p>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="bg-gray-900 rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-bold">📋 Incident Reports</h3>
                  <button
                    type="button"
                    onClick={loadIncidents}
                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition"
                  >
                    Refresh
                  </button>
                </div>

                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {incidents.length === 0 ? (
                    <p className="text-gray-500 text-sm text-center py-4">No incidents reported yet</p>
                  ) : (
                    incidents.map((incident) => (
                      <div key={incident.id} className="bg-gray-800 p-4 rounded-lg border border-gray-700 incident-card">
                        <div className="flex items-center justify-between mb-3">
                          <div>
                            <h4 className="font-bold text-white">Incident {incident.id}</h4>
                            <p className="text-sm text-gray-400">{incident.timestamp}</p>
                          </div>
                          <span
                            className={`status-badge ${
                              incident.severity === 'HIGH'
                                ? 'bg-red-600'
                                : incident.severity === 'MEDIUM'
                                ? 'bg-yellow-600'
                                : 'bg-green-600'
                            }`}
                          >
                            {incident.severity}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm mb-3">
                          <div>
                            <p className="text-gray-400">Location</p>
                            <p className="text-white">{incident.location}</p>
                          </div>
                          <div>
                            <p className="text-gray-400">Confidence</p>
                            <p className="text-white">{(incident.confidence * 100).toFixed(1)}%</p>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button
                            onClick={() => generateIncidentPDF(incident.id)}
                            className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm py-2 rounded transition"
                          >
                            📄 PDF Report
                          </button>
                          <button
                            onClick={() => editIncidentNotes(incident.id)}
                            className="flex-1 bg-gray-600 hover:bg-gray-700 text-white text-sm py-2 rounded transition"
                          >
                            📝 Notes
                          </button>
                          <button
                            onClick={() => deleteIncident(incident.id)}
                            className="px-3 bg-red-600 hover:bg-red-700 text-white text-sm py-2 rounded transition"
                          >
                            🗑️
                          </button>
                        </div>
                        {incident.notes ? (
                          <div className="mt-3 p-2 bg-gray-700 rounded text-sm">
                            <p className="text-gray-300">{incident.notes}</p>
                          </div>
                        ) : null}
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="bg-gray-900 rounded-xl p-6">
                <h3 className="text-lg font-bold mb-4">Camera Status & Monitoring</h3>
                <div className="space-y-3">
                  {cameraDefinitions.length === 0 ? (
                    <p className="text-gray-500 text-sm text-center py-4">Loading camera information...</p>
                  ) : (
                    cameraDefinitions.map((cam) => (
                      <div key={cam.id} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-bold text-lg">{cam.name}</h4>
                          <span
                            className={`status-badge ${
                              cam.isLive ? (cam.status === 'FALL DETECTED' ? 'bg-red-600' : 'bg-green-600') : 'bg-gray-600'
                            }`}
                          >
                            {cam.isLive ? '● LIVE' : '○ OFFLINE'}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <p className="text-gray-400">Status</p>
                            <p className={`${cam.isLive ? 'text-green-500' : 'text-gray-500'} font-semibold`}>{cam.status || 'Offline'}</p>
                          </div>
                          <div>
                            <p className="text-gray-400">FPS</p>
                            <p className="text-white font-semibold">{cam.fps ? cam.fps.toFixed(1) : '0.0'}</p>
                          </div>
                          <div>
                            <p className="text-gray-400">Confidence</p>
                            <p className="text-white font-semibold">{cam.confidence_score ? (cam.confidence_score * 100).toFixed(1) + '%' : '0.0%'}</p>
                          </div>
                          <div>
                            <p className="text-gray-400">Source</p>
                            <p className="text-white font-semibold truncate">{cam.source}</p>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="bg-gray-900 rounded-xl p-6">
                <h3 className="text-lg font-bold mb-4">Add Camera Source</h3>
                <div className="flex gap-2 mb-4">
                  <button
                    onClick={() => setActiveTab('webcam')}
                    className={`flex-1 py-2 rounded-lg font-semibold ${activeTab === 'webcam' ? 'bg-blue-600' : 'bg-gray-700'}`}
                  >
                    Webcam/URL
                  </button>
                  <button
                    onClick={() => setActiveTab('upload')}
                    className={`flex-1 py-2 rounded-lg font-semibold ${activeTab === 'upload' ? 'bg-blue-600' : 'bg-gray-700'}`}
                  >
                    Video File
                  </button>
                </div>

                {activeTab === 'webcam' ? (
                  <form onSubmit={addCamera} className="space-y-4">
                    <input
                      type="text"
                      placeholder="Camera Name (e.g., Living Room)"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                      value={cameraName}
                      onChange={(event) => setCameraName(event.target.value)}
                      required
                    />
                    <div>
                      <input
                        type="text"
                        placeholder="Source (0 for webcam, URL, or path)"
                        className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                        value={cameraSource}
                        onChange={(event) => setCameraSource(event.target.value)}
                        required
                      />
                      <p className="text-xs text-gray-500 mt-2">Examples: 0, 1, 2 or http://192.168.1.100:8080/video</p>
                    </div>
                    <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-lg transition">
                      Add Camera
                    </button>
                  </form>
                ) : (
                  <form onSubmit={submitUpload} className="space-y-4">
                    <input
                      type="text"
                      placeholder="Stream Name (e.g., Test Fall Video)"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                      value={uploadName}
                      onChange={(event) => setUploadName(event.target.value)}
                      required
                    />
                    <div className="w-full">
                      <label className="block text-sm font-medium mb-2 text-gray-300">Select Video File</label>
                      <input
                        type="file"
                        accept="video/*"
                        className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white file:cursor-pointer hover:file:bg-blue-700 cursor-pointer"
                        onChange={(event) => setUploadFile(event.target.files?.[0] || null)}
                        required
                      />
                      <p className="text-xs text-gray-500 mt-2">Supported: MP4, AVI, MOV, MKV (Max 500MB)</p>
                    </div>

                    {isUploading ? (
                      <div>
                        <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${uploadProgress.percent}%` }}
                          />
                        </div>
                        <p className="text-sm text-gray-400 text-center">{uploadProgress.text}</p>
                        <button
                          type="button"
                          onClick={cancelUpload}
                          className="mt-2 w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-2 rounded-lg transition"
                        >
                          Cancel Upload
                        </button>
                      </div>
                    ) : null}

                    <button
                      type="submit"
                      disabled={isUploading}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-lg transition disabled:opacity-60"
                    >
                      {isUploading ? 'Uploading...' : 'Upload & Start Stream'}
                    </button>
                  </form>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {showCameraManager ? (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-700 flex items-center justify-between">
              <h2 className="text-2xl font-bold">Manage Cameras</h2>
              <button onClick={() => setShowCameraManager(false)} className="text-gray-400 hover:text-white text-3xl leading-none">
                &times;
              </button>
            </div>
            <div className="p-6">
              {cameraDefinitions.length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-4">Loading...</p>
              ) : (
                <div className="space-y-2">
                  {cameraDefinitions.map((cam) => (
                    <div key={cam.id} className="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
                      <div className="flex-1">
                        <p className="font-semibold">{cam.name}</p>
                        <p className="text-xs text-gray-500">Source: {cam.source}</p>
                        <p className={`text-xs ${cam.isLive ? 'text-green-500' : 'text-gray-500'}`}>
                          {cam.isLive ? '● Live' : '○ Stopped'}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        {cam.isLive ? (
                          <button
                            onClick={() => stopCamera(cam.id)}
                            className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition"
                          >
                            Stop
                          </button>
                        ) : (
                          <button
                            onClick={() => startCamera(cam.id)}
                            className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition"
                          >
                            Start
                          </button>
                        )}
                        <button
                          onClick={() => removeCamera(cam.id)}
                          className="px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded text-sm font-medium transition"
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {showTelegramSubscribers ? (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-700 flex items-center justify-between sticky top-0 bg-gray-800 z-10">
              <h2 className="text-2xl font-bold">Telegram Subscribers</h2>
              <button onClick={() => setShowTelegramSubscribers(false)} className="text-gray-400 hover:text-white text-3xl leading-none">
                &times;
              </button>
            </div>
            <div className="p-6 space-y-6">
              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="font-semibold mb-3">How to add subscribers:</h3>
                <ul className="space-y-2 text-sm text-gray-300">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400 font-bold mt-0.5">1.</span>
                    <span>Users can send <span className="bg-gray-800 px-1 py-0.5 rounded text-yellow-300">/start</span> to your bot (automatic)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400 font-bold mt-0.5">2.</span>
                    <span>Or manually add their Chat ID below</span>
                  </li>
                </ul>
              </div>

              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="font-semibold mb-3">Add Manually</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium mb-2">Chat ID</label>
                    <input
                      type="text"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                      value={manualChatId}
                      onChange={(event) => setManualChatId(event.target.value)}
                      placeholder="e.g., 123456789"
                    />
                    <p className="text-xs text-gray-500 mt-1">User's unique Telegram ID (numbers only)</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Name (optional)</label>
                    <input
                      type="text"
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
                      value={manualName}
                      onChange={(event) => setManualName(event.target.value)}
                      placeholder="e.g., John's Phone"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={addManualSubscriber}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition"
                  >
                    Add Subscriber
                  </button>
                </div>
              </div>

              <div className="bg-gray-900 rounded-lg p-4">
                <h3 className="font-semibold mb-3">Active Subscribers</h3>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {adminSubscribers.length === 0 ? (
                    <p className="text-gray-500 text-sm text-center py-4">No subscribers yet. Ask users to send /start to your bot!</p>
                  ) : (
                    adminSubscribers.map((sub) => (
                      <div key={sub.chat_id} className="bg-gray-800 p-3 rounded-lg flex items-center justify-between">
                        <div className="flex-1">
                          <p className="font-medium">{sub.name || 'Unknown'}</p>
                          <p className="text-xs text-gray-400">ID: {sub.chat_id}</p>
                          {sub.username ? <p className="text-xs text-gray-400">@{sub.username}</p> : null}
                        </div>
                        <button
                          type="button"
                          onClick={() => removeSubscriber(sub.chat_id)}
                          className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition"
                        >
                          Remove
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <div id="toast-container" className="fixed bottom-4 right-4 z-50 space-y-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`text-white px-6 py-3 rounded-lg shadow-xl fade-in ${
              toast.type === 'success'
                ? 'bg-green-600'
                : toast.type === 'error'
                ? 'bg-red-600'
                : toast.type === 'warning'
                ? 'bg-yellow-600'
                : 'bg-blue-600'
            }`}
          >
            {toast.message}
          </div>
        ))}
      </div>
    </div>
  )
}

export default DashboardPage
