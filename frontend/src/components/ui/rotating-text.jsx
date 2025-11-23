import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

export const RotatingText = ({ texts, className }) => {
  const [index, setIndex] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prevIndex) => (prevIndex + 1) % texts.length)
    }, 2500)

    return () => clearInterval(interval)
  }, [texts.length])

  return (
    <div className="relative inline-block">
      <AnimatePresence mode="wait">
        <motion.span
          key={index}
          initial={{ rotateX: 90, opacity: 0 }}
          animate={{ rotateX: 0, opacity: 1 }}
          exit={{ rotateX: -90, opacity: 0 }}
          transition={{
            duration: 0.5,
            ease: "easeInOut",
          }}
          className={className}
          style={{
            transformOrigin: 'center center',
            display: 'inline-block',
          }}
        >
          {texts[index]}
        </motion.span>
      </AnimatePresence>
    </div>
  )
}
