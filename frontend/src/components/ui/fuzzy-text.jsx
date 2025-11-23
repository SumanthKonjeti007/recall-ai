import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

export const FuzzyText = ({ text, className }) => {
  const svgRef = useRef(null)

  useEffect(() => {
    if (!svgRef.current) return

    const filter = svgRef.current.querySelector('feTurbulence')
    if (!filter) return

    let frame = 0
    const animate = () => {
      const baseFrequency = 0.015 + Math.sin(frame * 0.05) * 0.005
      filter.setAttribute('baseFrequency', `${baseFrequency} ${baseFrequency}`)
      frame++
      requestAnimationFrame(animate)
    }

    animate()
  }, [])

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        className="absolute inset-0 w-0 h-0"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <filter id="fuzzy-text">
            <feTurbulence
              type="turbulence"
              baseFrequency="0.02"
              numOctaves="3"
              result="noise"
              seed="2"
            />
            <feDisplacementMap
              in="SourceGraphic"
              in2="noise"
              scale="6"
              xChannelSelector="R"
              yChannelSelector="G"
            />
          </filter>
        </defs>
      </svg>

      <motion.span
        initial={{ opacity: 0, filter: 'url(#fuzzy-text) blur(10px)' }}
        animate={{ opacity: 1, filter: 'url(#fuzzy-text) blur(0px)' }}
        transition={{ duration: 1.5, delay: 0.5 }}
        className={className}
        style={{ filter: 'url(#fuzzy-text)' }}
      >
        {text}
      </motion.span>
    </div>
  )
}
