import { motion } from 'framer-motion'
import { RotatingText } from './ui/rotating-text'
import { Brain, Sparkles, Zap, Search } from 'lucide-react'

export const LandingPage = ({ onEnter }) => {
  const rotatingWords = ['Recall', 'Remember', 'Retrieve']

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.6 }}
      onClick={onEnter}
      className="min-h-screen bg-[#0a0a0a] text-white flex items-center justify-center relative overflow-hidden cursor-pointer group"
    >
      {/* Subtle Gradient Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-br from-orange-950/20 via-purple-950/20 to-blue-950/20" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,transparent_0%,#0a0a0a_100%)]" />

        <motion.div
          animate={{
            opacity: [0.3, 0.5, 0.3],
            scale: [1, 1.1, 1],
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-r from-orange-500/10 via-purple-500/10 to-blue-500/10 rounded-full blur-3xl"
        />
      </div>

      {/* Content */}
      <div className="relative z-10 text-center space-y-12 px-4">
        {/* Logo */}
        <motion.div
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ duration: 0.8, delay: 0.2, type: "spring", stiffness: 200 }}
          className="flex justify-center"
        >
          <div className="relative">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="absolute -inset-4 bg-gradient-to-r from-orange-500/20 via-purple-500/20 to-blue-500/20 rounded-full blur-2xl"
            />

            <div className="relative bg-gradient-to-br from-gray-900 to-black p-6 rounded-full border border-gray-800/50">
              <Brain className="w-16 h-16 text-purple-400" />
            </div>
          </div>
        </motion.div>

        {/* Title with Rotating Text Effect */}
        <div className="space-y-4">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="text-7xl md:text-9xl font-bold tracking-tight"
          >
            <RotatingText
              texts={rotatingWords}
              className="bg-gradient-to-r from-orange-400 via-purple-400 to-blue-400 bg-clip-text text-transparent"
            />
          </motion.h1>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 1.2 }}
            className="text-xl md:text-2xl text-gray-500 font-light"
          >
            Your AI-Powered Memory Assistant
          </motion.p>
        </div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.5 }}
          className="flex flex-wrap justify-center gap-4"
        >
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-gray-900/30 border border-gray-800/50 backdrop-blur-sm"
          >
            <Search className="w-4 h-4 text-orange-400" />
            <span className="text-sm text-gray-400">Hybrid RAG</span>
          </motion.div>
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-gray-900/30 border border-gray-800/50 backdrop-blur-sm"
          >
            <Sparkles className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-400">Vector Search</span>
          </motion.div>
          <motion.div
            whileHover={{ scale: 1.05 }}
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-gray-900/30 border border-gray-800/50 backdrop-blur-sm"
          >
            <Zap className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-gray-400">Lightning Fast</span>
          </motion.div>
        </motion.div>
      </div>

      {/* Subtle particle effects */}
      {[...Array(15)].map((_, i) => (
        <motion.div
          key={i}
          initial={{
            x: Math.random() * window.innerWidth,
            y: window.innerHeight + 20,
            opacity: 0,
          }}
          animate={{
            y: -20,
            opacity: [0, 0.3, 0],
          }}
          transition={{
            duration: Math.random() * 15 + 10,
            repeat: Infinity,
            delay: Math.random() * 10,
            ease: "linear",
          }}
          className="absolute w-1 h-1 bg-purple-500/50 rounded-full"
        />
      ))}
    </motion.div>
  )
}
