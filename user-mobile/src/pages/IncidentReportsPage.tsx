import { useEffect, useState } from 'react'

type Incident = {
  id: string
  timestamp: string
  severity: 'HIGH' | 'MEDIUM' | 'LOW'
  stream_name?: string
  camera_name?: string
  location: string
  confidence: number
  notes?: string
}

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

const localIncidents: Incident[] = [
  {
    id: 'INC-1001',
    timestamp: '2026-07-08 09:15:00',
    severity: 'HIGH',
    stream_name: 'Main Hall Stream',
    camera_name: 'Main Hall Camera',
    location: 'Main Hall',
    confidence: 0.94,
    notes: 'Local sample incident for offline mode.',
  },
]

const toIncidentEpoch = (timestamp: string) => {
  const normalized = timestamp.replace(' ', 'T')
  const parsed = Date.parse(normalized)
  return Number.isNaN(parsed) ? 0 : parsed
}

const sortIncidentsNewestFirst = (incidentList: Incident[]) => {
  return [...incidentList].sort((a, b) => toIncidentEpoch(b.timestamp) - toIncidentEpoch(a.timestamp))
}

function getSeverityClass(severity: Incident['severity']) {
  if (severity === 'HIGH') return 'severity-high'
  if (severity === 'MEDIUM') return 'severity-medium'
  return 'severity-low'
}

function IncidentReportsPage() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [openingPdfId, setOpeningPdfId] = useState<string | null>(null)

  const openIncidentPdf = (incidentId: string) => {
    if (openingPdfId === incidentId) {
      return
    }

    setOpeningPdfId(incidentId)
    const pdfUrl = `${getServerBaseUrl()}/api/incidents/${incidentId}/pdf`
    const link = document.createElement('a')
    link.href = pdfUrl
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    document.body.appendChild(link)
    link.click()
    link.remove()

    window.setTimeout(() => {
      setOpeningPdfId((current) => (current === incidentId ? null : current))
    }, 1200)
  }

  const loadIncidents = async () => {
    try {
      const response = await fetch(`${getServerBaseUrl()}/api/incidents`, {
        method: 'GET',
        cache: 'no-store',
      })
      if (!response.ok) {
        throw new Error(`incidents failed with status ${response.status}`)
      }

      const data = await response.json()
      const incidentList = Array.isArray(data.incidents) ? data.incidents : []
      setIncidents(sortIncidentsNewestFirst(incidentList))
    } catch {
      setIncidents(sortIncidentsNewestFirst(localIncidents))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadIncidents()
    const timer = window.setInterval(loadIncidents, 8000)

    return () => window.clearInterval(timer)
  }, [])

  return (
    <div className="page-shell">
      <div className="content-wrap">
        <header className="panel glass">
          <div className="brand-row">
            <div className="brand-dot" aria-hidden="true">
              <img src="/static/images/fallguard-logo.png" alt="" className="brand-logo" />
            </div>
            <div>
              <h1 className="title">Incident Reports</h1>
              <p className="subtitle">Live records of detected falls</p>
            </div>
          </div>
        </header>

        <section className="panel glass incident-list">
          {isLoading ? <p className="muted">Loading incident reports...</p> : null}
          {!isLoading && incidents.length === 0 ? (
            <p className="muted">No incident reports yet.</p>
          ) : null}

          {incidents.map((incident) => (
            <article className="incident-card" key={incident.id}>
              <div className="incident-header">
                <strong>{incident.id}</strong>
                <span className={`severity-pill ${getSeverityClass(incident.severity)}`}>
                  {incident.severity}
                </span>
              </div>
              <p className="incident-detail"><strong>Time:</strong> {incident.timestamp}</p>
              <p className="incident-detail"><strong>Stream:</strong> {incident.stream_name || '-'}</p>
              <p className="incident-detail"><strong>Camera:</strong> {incident.camera_name || '-'}</p>
              <p className="incident-detail"><strong>Location:</strong> {incident.location}</p>
              <p className="incident-detail">
                <strong>Confidence:</strong> {(incident.confidence * 100).toFixed(1)}%
              </p>
              <div className="incident-actions">
                <button
                  type="button"
                  className="incident-pdf-btn"
                  onClick={() => openIncidentPdf(incident.id)}
                  disabled={openingPdfId === incident.id}
                >
                  {openingPdfId === incident.id ? 'Opening...' : 'Open PDF'}
                </button>
              </div>
              {incident.notes ? <p className="incident-note">{incident.notes}</p> : null}
            </article>
          ))}
        </section>
      </div>
    </div>
  )
}

export default IncidentReportsPage
