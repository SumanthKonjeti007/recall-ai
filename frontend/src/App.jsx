import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import { LandingPage } from './components/LandingPage'
import { MainApp } from './components/MainApp'

function App() {
  const [showLanding, setShowLanding] = useState(true)

  const handleEnter = () => {
    setShowLanding(false)
  }

  return (
    <AnimatePresence mode="wait">
      {showLanding ? (
        <LandingPage key="landing" onEnter={handleEnter} />
      ) : (
        <MainApp key="main" />
      )}
    </AnimatePresence>
  )
}

export default App
