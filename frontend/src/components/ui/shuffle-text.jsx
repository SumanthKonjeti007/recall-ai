import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'

export const ShuffleText = ({ text, className }) => {
  const [displayText, setDisplayText] = useState(text)
  const iterations = useRef(0)
  const intervalRef = useRef(null)

  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

  useEffect(() => {
    const shuffle = () => {
      iterations.current = 0

      clearInterval(intervalRef.current)

      intervalRef.current = setInterval(() => {
        setDisplayText(
          text
            .split("")
            .map((letter, index) => {
              if (index < iterations.current) {
                return text[index]
              }
              return letters[Math.floor(Math.random() * 26)]
            })
            .join("")
        )

        if (iterations.current >= text.length) {
          clearInterval(intervalRef.current)
        }

        iterations.current += 1 / 3
      }, 30)
    }

    // Start shuffle after mount
    const timeout = setTimeout(shuffle, 500)

    return () => {
      clearTimeout(timeout)
      clearInterval(intervalRef.current)
    }
  }, [text])

  return (
    <motion.span
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={className}
    >
      {displayText}
    </motion.span>
  )
}
