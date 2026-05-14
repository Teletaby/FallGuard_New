import { useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

function LandingPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()

  const activePage = useMemo(() => {
    const page = searchParams.get('page')
    return page === 'learn-more' ? 'learn-more' : 'landing'
  }, [searchParams])

  const goToLiveFeed = () => navigate('/')
  const goToHomepage = () => navigate('/landing')
  const goToLearnMore = () => navigate('/landing?page=learn-more')

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {activePage === 'landing' ? (
        <div id="landing-page" className="hero-section gradient-bg">
          <nav className="absolute top-0 left-0 right-0 z-20 px-4 py-4 sm:px-6 bg-gradient-to-b from-black/50 to-transparent">
            <div className="container mx-auto flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <img
                  id="logo-image"
                  src="/static/images/fallguard-logo.svg"
                  alt="FallGuard Logo"
                  className="w-10 h-10"
                />
                <span className="text-lg sm:text-xl font-bold text-white">FallGuard</span>
              </div>
              <div className="flex items-center gap-2 sm:gap-4">
                <button
                  onClick={goToLearnMore}
                  className="text-sm font-semibold hover:text-blue-400 transition"
                >
                  Learn More
                </button>
                <button
                  onClick={goToLiveFeed}
                  className="glass-effect px-3 py-2 sm:px-6 rounded-full text-sm font-semibold hover:bg-white/20 transition flex items-center gap-2"
                >
                  <svg className="hidden sm:block w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                    />
                  </svg>
                  Sign In
                </button>
              </div>
            </div>
          </nav>

          <div className="container mx-auto px-4 sm:px-6 py-28 sm:py-32 relative z-10">
            <div className="text-center max-w-5xl mx-auto fade-in">
              <div className="mb-8 sm:mb-12">
                <svg className="w-20 h-20 sm:w-28 sm:h-28 md:w-32 md:h-32 mx-auto text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
              </div>

              <h1 className="text-4xl sm:text-5xl md:text-7xl font-bold mb-6 leading-tight text-white drop-shadow-2xl">
                FallGuard: Protect Your Loved Ones
              </h1>

              <p className="text-lg sm:text-xl md:text-2xl text-white/90 mb-10 sm:mb-12 max-w-3xl mx-auto font-light leading-relaxed">
                A lightweight fall detection system
                <br />
                for ultimate peace of mind and
                <br />
                keeping your family safe.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 sm:gap-6 justify-center">
                <button
                  onClick={goToLearnMore}
                  className="group relative inline-flex items-center justify-center gap-3 bg-blue-500 text-white px-8 py-4 rounded-lg text-base sm:text-lg font-bold hover:bg-blue-600 transition-all duration-300 transform hover:scale-105 shadow-2xl w-full sm:w-auto"
                >
                  <span>LEARN MORE</span>
                  <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {activePage === 'learn-more' ? (
        <div id="learn-more-page" className="min-h-screen bg-gray-900 text-white">
          <nav className="sticky top-0 z-20 px-4 py-4 sm:px-6 bg-gray-800 shadow-lg">
            <div className="container mx-auto flex items-center justify-between gap-3">
              <button onClick={goToHomepage} className="flex items-center gap-3 hover:opacity-80 transition min-w-0">
                <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <span className="text-lg sm:text-xl font-bold truncate">FallGuard</span>
              </button>
              <button onClick={goToLiveFeed} className="bg-blue-500 hover:bg-blue-600 px-4 sm:px-6 py-2 rounded-lg font-semibold transition whitespace-nowrap">
                Sign In
              </button>
            </div>
          </nav>

          <div className="bg-gradient-to-b from-gray-800 to-gray-900 py-14 sm:py-20">
            <div className="container mx-auto px-4 sm:px-6">
              <h2 className="text-3xl sm:text-4xl md:text-6xl font-bold mb-6">Peace of mind for your family</h2>
              <p className="text-lg sm:text-xl text-gray-300 max-w-2xl">
                Advanced AI-powered fall detection that keeps your loved ones safe, automatically.
              </p>
            </div>
          </div>

          <div className="py-14 sm:py-20 px-4 sm:px-6">
            <div className="container mx-auto">
              <h3 className="text-3xl sm:text-4xl font-bold mb-10 sm:mb-16 text-center">What We Offer:</h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
                <div className="bg-gray-800 rounded-xl p-8 hover:bg-gray-700 transition h-full">
                  <div className="text-5xl mb-6">⚡</div>
                  <h4 className="text-2xl font-bold mb-4">Efficient & Agile</h4>
                  <p className="text-gray-300 text-lg">
                    We offer a lightweight fall detection system that can run smoothly and efficiently on your system
                  </p>
                </div>

                <div className="bg-gray-800 rounded-xl p-8 hover:bg-gray-700 transition h-full">
                  <div className="text-5xl mb-6">🔔</div>
                  <h4 className="text-2xl font-bold mb-4">Instant Alerts</h4>
                  <p className="text-gray-300 text-lg">
                    Get real-time notifications via Telegram when a fall is detected. Stay informed instantly.
                  </p>
                </div>

                <div className="bg-gray-800 rounded-xl p-8 hover:bg-gray-700 transition h-full">
                  <div className="text-5xl mb-6">🚨</div>
                  <h4 className="text-2xl font-bold mb-4">Emergency SOS</h4>
                  <p className="text-gray-300 text-lg">
                    Rapid response system with emergency alerts and detailed fall detection confidence scores.
                  </p>
                </div>
              </div>

              <div className="bg-blue-600 rounded-xl p-6 sm:p-12 mb-16">
                <h3 className="text-2xl sm:text-3xl font-bold mb-8">Why Choose FallGuard?</h3>
                <ul className="space-y-4 text-base sm:text-lg">
                  <li className="flex items-start sm:items-center gap-4">
                    <svg className="w-6 h-6 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span>AI-powered accuracy with real-time detection</span>
                  </li>
                  <li className="flex items-start sm:items-center gap-4">
                    <svg className="w-6 h-6 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span>Multi-camera support for comprehensive coverage</span>
                  </li>
                  <li className="flex items-start sm:items-center gap-4">
                    <svg className="w-6 h-6 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span>Instant Telegram notifications for peace of mind</span>
                  </li>
                  <li className="flex items-start sm:items-center gap-4">
                    <svg className="w-6 h-6 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span>Lightweight and efficient system</span>
                  </li>
                  <li className="flex items-start sm:items-center gap-4">
                    <svg className="w-6 h-6 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span>Privacy-first design with multiple viewing modes</span>
                  </li>
                  <li className="flex items-start sm:items-center gap-4">
                    <svg className="w-6 h-6 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    <span>Comprehensive incident reporting and PDF exports</span>
                  </li>
                </ul>
              </div>

              <div className="text-center">
                <button
                  onClick={goToLiveFeed}
                  className="bg-blue-500 hover:bg-blue-600 px-8 sm:px-12 py-4 rounded-lg font-bold text-lg sm:text-xl transition transform hover:scale-105 w-full sm:w-auto"
                >
                  Get Started Now
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default LandingPage
