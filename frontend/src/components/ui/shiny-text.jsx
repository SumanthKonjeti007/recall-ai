import { motion } from 'framer-motion'
import { cn } from '../../lib/utils'

export const ShinyText = ({ text, className }) => {
  return (
    <motion.span
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8 }}
      className={cn(
        "inline-block bg-gradient-to-r from-orange-400 via-purple-400 to-blue-400 bg-clip-text text-transparent animate-gradient bg-[length:200%_auto]",
        className
      )}
    >
      {text}
    </motion.span>
  )
}
