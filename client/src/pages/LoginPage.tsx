import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

const ADMIN_AUTH_KEY = 'fallguard_admin_authenticated'
const ADMIN_PASSWORD = 'admin'

function LoginPage() {
  const navigate = useNavigate()
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isChecking, setIsChecking] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)

  useEffect(() => {
    const isAuthenticated = window.localStorage.getItem(ADMIN_AUTH_KEY) === 'true'
    if (isAuthenticated) {
      navigate('/dashboard', { replace: true })
      return
    }

    setIsChecking(false)
  }, [navigate])

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError('')
    setIsSubmitting(true)

    try {
      if (password !== ADMIN_PASSWORD) {
        setError('Invalid password')
        setPassword('')
        return
      }

      window.localStorage.setItem(ADMIN_AUTH_KEY, 'true')
      navigate('/dashboard')
    } catch {
      setError('Login failed')
    } finally {
      setIsSubmitting(false)
    }
  }

  if (isChecking) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900 text-white">
        <p className="text-gray-400">Checking session...</p>
      </div>
    )
  }

  return (
    <div className="bg-gray-900 text-white min-h-screen flex items-center justify-center">
      <div className="w-full max-w-md mx-auto bg-gray-800 rounded-2xl shadow-2xl p-8">
        <div className="flex items-center justify-center gap-3 mb-6">
          <img
            src="/static/images/fallguard-logo.svg"
            alt="FallGuard Logo"
            className="w-12 h-12"
          />
          <span className="text-2xl font-bold">FallGuard</span>
        </div>
        <h1 className="text-xl font-semibold text-center mb-6">Admin Login</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="password" className="block text-sm font-medium mb-2">
              Password
            </label>
            <input
              type="password"
              id="password"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 outline-none"
              placeholder="Enter admin password"
              required
              autoFocus
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </div>
          <button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-lg transition disabled:opacity-60"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Checking...' : 'Login'}
          </button>
          {error ? (
            <p className="text-sm text-red-400 text-center">{error}</p>
          ) : null}
        </form>
        <div className="mt-6 text-center text-sm text-gray-400">
          Looking for the public landing page?
          <Link to="/landing" className="text-blue-400 hover:text-blue-300 ml-1">
            View landing
          </Link>
        </div>
      </div>
    </div>
  )
}

export default LoginPage
