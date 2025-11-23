import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Plus, Sparkles, Zap, Database, TrendingUp, Cpu, User, Bot, Calendar, MapPin, Users, ShoppingBag, Github, HelpCircle } from 'lucide-react'

export const MainApp = () => {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([]) // Chat messages array
  const [loading, setLoading] = useState(false)
  const inputRef = useRef(null)
  const messagesEndRef = useRef(null)

  const suggestionCards = [
    {
      icon: Calendar,
      title: "Timeline Queries",
      description: "Search by specific dates or time periods",
      example: "What did I do last weekend?"
    },
    {
      icon: MapPin,
      title: "Location Activities",
      description: "Explore places and destinations",
      example: "Which restaurants did I visit in Paris?"
    },
    {
      icon: Users,
      title: "People & Contacts",
      description: "Find interactions and connections",
      example: "Show me messages from Sarah"
    },
    {
      icon: ShoppingBag,
      title: "Service Requests",
      description: "Track activities and preferences",
      example: "What are my favorite cuisines?"
    }
  ]

  useEffect(() => {
    // Load chat history from localStorage
    const savedMessages = localStorage.getItem('recall-messages')
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages))
    }
  }, [])

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    const userMessage = {
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
      id: Date.now()
    }

    // Add user message to chat
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setQuery('')
    setLoading(true)

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.content })
      })

      const data = await response.json()

      const assistantMessage = {
        role: 'assistant',
        content: data.answer,
        metadata: data.metadata,
        timestamp: new Date().toISOString(),
        id: Date.now() + 1
      }

      const newMessages = [...updatedMessages, assistantMessage]
      setMessages(newMessages)
      localStorage.setItem('recall-messages', JSON.stringify(newMessages))
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        error: true,
        timestamp: new Date().toISOString(),
        id: Date.now() + 1
      }
      const newMessages = [...updatedMessages, errorMessage]
      setMessages(newMessages)
    } finally {
      setLoading(false)
    }
  }

  const handleSuggestionClick = (example) => {
    setQuery(example)
    inputRef.current?.focus()
  }

  const handleNewChat = () => {
    setMessages([])
    setQuery('')
    localStorage.removeItem('recall-messages')
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="min-h-screen bg-[#0a0a0a] text-white flex flex-col"
    >
      {/* Header */}
      <header className="border-b border-gray-800/50 backdrop-blur-sm bg-[#0a0a0a]/80 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg blur opacity-50" />
              <div className="relative bg-gradient-to-br from-purple-600 to-blue-600 p-2 rounded-lg">
                <Sparkles className="w-5 h-5" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-bold">Recall</h1>
              <p className="text-xs text-gray-500">AI Memory Assistant</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg hover:bg-gray-800/50 transition-colors"
            >
              <Github className="w-5 h-5 text-gray-400" />
            </a>
            <button className="p-2 rounded-lg hover:bg-gray-800/50 transition-colors">
              <HelpCircle className="w-5 h-5 text-gray-400" />
            </button>
            <button
              onClick={handleNewChat}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-purple-500/10 to-blue-500/10 hover:from-purple-500/20 hover:to-blue-500/20 border border-gray-800 transition-colors"
            >
              <Plus className="w-4 h-4" />
              <span className="text-sm">New Chat</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-8">
          {messages.length === 0 ? (
            /* Welcome Screen */
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="space-y-12"
            >
              {/* Hero Section */}
              <div className="text-center space-y-6 py-12">
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.8, delay: 0.2 }}
                  className="inline-block"
                >
                  <div className="relative">
                    <div className="absolute -inset-4 bg-gradient-to-r from-purple-500 via-blue-500 to-cyan-500 rounded-full blur-2xl opacity-30 animate-pulse" />
                    <div className="relative bg-gradient-to-br from-purple-500 to-blue-600 p-8 rounded-3xl">
                      <Sparkles className="w-16 h-16" />
                    </div>
                  </div>
                </motion.div>

                <div className="space-y-3">
                  <h2 className="text-5xl font-bold bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
                    Hello! I'm Recall
                  </h2>
                  <p className="text-xl text-gray-400">
                    Your AI-powered memory assistant for curated experiences
                  </p>
                </div>
              </div>

              {/* Suggestion Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {suggestionCards.map((card, idx) => (
                  <motion.button
                    key={idx}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.4 + idx * 0.1 }}
                    onClick={() => handleSuggestionClick(card.example)}
                    className="group relative p-6 rounded-2xl bg-gradient-to-br from-gray-900/50 to-gray-800/30 border border-gray-800/50 hover:border-gray-700 transition-all text-left overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-br from-purple-500/0 via-blue-500/0 to-cyan-500/0 group-hover:from-purple-500/5 group-hover:via-blue-500/5 group-hover:to-cyan-500/5 transition-all duration-300" />

                    <div className="relative space-y-3">
                      <div className="flex items-start justify-between">
                        <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/20">
                          <card.icon className="w-6 h-6 text-purple-400" />
                        </div>
                      </div>

                      <div className="space-y-1">
                        <h3 className="font-semibold text-white group-hover:text-purple-300 transition-colors">
                          {card.title}
                        </h3>
                        <p className="text-sm text-gray-500">
                          {card.description}
                        </p>
                      </div>

                      <div className="pt-2 border-t border-gray-800/50">
                        <p className="text-xs text-gray-600 italic">
                          "{card.example}"
                        </p>
                      </div>
                    </div>
                  </motion.button>
                ))}
              </div>
            </motion.div>
          ) : (
            /* Chat Messages */
            <div className="space-y-6 pb-32">
              <AnimatePresence>
                {messages.map((message, idx) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {message.role === 'assistant' && (
                      <div className="flex-shrink-0">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center">
                          <Bot className="w-5 h-5" />
                        </div>
                      </div>
                    )}

                    <div className={`max-w-3xl ${message.role === 'user' ? 'bg-gradient-to-br from-purple-600/20 to-blue-600/20 border-purple-500/30' : 'bg-[#1a1a1a] border-gray-800'} rounded-2xl border p-6 space-y-4`}>
                      {message.role === 'assistant' && message.metadata && (
                        <div className="flex flex-wrap gap-2 pb-4 border-b border-gray-800">
                          {message.metadata.route && (
                            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-purple-500/10 border border-purple-500/20">
                              <Database className="w-3 h-3 text-purple-400" />
                              <span className="text-xs text-purple-300">{message.metadata.route}</span>
                            </div>
                          )}
                          {message.metadata.confidence && (
                            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg ${
                              message.metadata.confidence === 'high'
                                ? 'bg-green-500/10 border border-green-500/20'
                                : message.metadata.confidence === 'medium'
                                ? 'bg-yellow-500/10 border border-yellow-500/20'
                                : 'bg-orange-500/10 border border-orange-500/20'
                            }`}>
                              <TrendingUp className={`w-3 h-3 ${
                                message.metadata.confidence === 'high' ? 'text-green-400' :
                                message.metadata.confidence === 'medium' ? 'text-yellow-400' : 'text-orange-400'
                              }`} />
                              <span className={`text-xs ${
                                message.metadata.confidence === 'high' ? 'text-green-300' :
                                message.metadata.confidence === 'medium' ? 'text-yellow-300' : 'text-orange-300'
                              }`}>
                                {message.metadata.confidence}
                              </span>
                            </div>
                          )}
                          {message.metadata.processing_time_ms && (
                            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-blue-500/10 border border-blue-500/20">
                              <Zap className="w-3 h-3 text-blue-400" />
                              <span className="text-xs text-blue-300">
                                {(message.metadata.processing_time_ms / 1000).toFixed(1)}s
                              </span>
                            </div>
                          )}
                          {message.metadata.sources_count !== undefined && (
                            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                              <Sparkles className="w-3 h-3 text-cyan-400" />
                              <span className="text-xs text-cyan-300">
                                {message.metadata.sources_count} sources
                              </span>
                            </div>
                          )}
                        </div>
                      )}

                      <div className="prose prose-invert max-w-none">
                        <div className={message.role === 'user' ? 'text-white' : ''}>
                          {message.content?.split('\n').map((line, i) => (
                            <p key={i} className={i > 0 ? 'mt-2' : ''}>{line}</p>
                          ))}
                        </div>
                      </div>

                      {message.role === 'assistant' && message.metadata?.sources && message.metadata.sources.length > 0 && (
                        <details className="pt-4 border-t border-gray-800 group">
                          <summary className="text-sm font-semibold text-gray-400 cursor-pointer flex items-center gap-2 hover:text-gray-300">
                            <Database className="w-4 h-4" />
                            View {message.metadata.sources_count || message.metadata.sources.length} Sources
                          </summary>
                          <div className="mt-3 space-y-2">
                            {message.metadata.sources.slice(0, 5).map((source, idx) => (
                              <div key={idx} className="bg-gray-900/50 p-3 rounded-lg text-sm">
                                {source.text ? (
                                  <p className="text-gray-400">{source.text}</p>
                                ) : (
                                  <div className="flex justify-between text-gray-500">
                                    <span>Source {idx + 1}</span>
                                    <span>Score: {(source.score * 100).toFixed(1)}%</span>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </details>
                      )}

                      <div className="flex items-center gap-2 pt-2 text-xs text-gray-600">
                        {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>

                    {message.role === 'user' && (
                      <div className="flex-shrink-0">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 to-pink-500 flex items-center justify-center">
                          <User className="w-5 h-5" />
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>

              {loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex gap-4"
                >
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center">
                      <Bot className="w-5 h-5" />
                    </div>
                  </div>
                  <div className="bg-[#1a1a1a] rounded-2xl border border-gray-800 p-6">
                    <div className="flex gap-2">
                      <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      {/* Fixed Input Bar */}
      <div className="sticky bottom-0 border-t border-gray-800/50 backdrop-blur-sm bg-[#0a0a0a]/80">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <form onSubmit={handleSubmit} className="relative">
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 via-blue-500 to-cyan-500 rounded-2xl opacity-20 group-hover:opacity-30 blur transition-opacity" />

              <div className="relative bg-[#1a1a1a] rounded-2xl border border-gray-800 focus-within:border-purple-500/50 transition-colors">
                <div className="flex items-center gap-3 p-4">
                  <input
                    ref={inputRef}
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask me anything..."
                    className="flex-1 bg-transparent text-white placeholder-gray-500 focus:outline-none"
                    disabled={loading}
                  />

                  <button
                    type="submit"
                    disabled={!query.trim() || loading}
                    className="p-3 rounded-xl bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Send className="w-5 h-5" />
                    )}
                  </button>
                </div>

                <div className="px-4 pb-3 text-xs text-gray-600">
                  Press <kbd className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">Enter</kbd> to send • <kbd className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">Shift + Enter</kbd> for new line
                </div>
              </div>
            </div>
          </form>
        </div>
      </div>
    </motion.div>
  )
}
