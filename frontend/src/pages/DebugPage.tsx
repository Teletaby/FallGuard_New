import { useState } from 'react'

function DebugPage() {
  const [output, setOutput] = useState<string[]>([])

  const log = (message: string) => {
    setOutput((prev) => [...prev, message])
    console.log(message)
  }

  const testModal = () => {
    log('Testing modal opening...')
    log('✓ Basic JavaScript is working')
    log('Open the admin panel from the dashboard to validate modal rendering.')
  }

  const testAPI = async () => {
    log('Testing API connection...')
    try {
      const response = await fetch('/api/admin/check')
      const data = await response.json()
      log(`✓ API Response: ${JSON.stringify(data)}`)
    } catch (error) {
      log(`✗ API Error: ${(error as Error).message}`)
    }
  }

  return (
    <div className="min-h-screen bg-white text-black p-8">
      <h1 className="text-2xl font-bold mb-4">FallGuard Debug Page</h1>
      <div className="flex gap-3 mb-6">
        <button onClick={testModal} className="px-4 py-2 bg-gray-200 rounded">
          Test Admin Panel
        </button>
        <button onClick={testAPI} className="px-4 py-2 bg-gray-200 rounded">
          Test API Connection
        </button>
      </div>
      <div className="bg-gray-100 p-4 rounded">
        {output.length === 0 ? <p>No debug output yet.</p> : null}
        {output.map((line, index) => (
          <p key={`${line}-${index}`}>{line}</p>
        ))}
      </div>
    </div>
  )
}

export default DebugPage
