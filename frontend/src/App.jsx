import { useState } from 'react'
import { Button } from './components/ui/button'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './components/ui/card'
import { Search, Brain, Sparkles } from 'lucide-react'

function App() {
  const [query, setQuery] = useState('')
  const [answer, setAnswer] = useState(null)
  const [loading, setLoading] = useState(false)

  const exampleQueries = [
    "What Italian restaurants did I visit?",
    "How many messages about family?",
    "Show me activities from last month",
    "What did I do related to travel?"
  ]

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setAnswer(null)

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      })

      const data = await response.json()
      setAnswer(data)
    } catch (error) {
      setAnswer({
        error: true,
        message: 'Failed to get answer. Please try again.'
      })
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (exampleQuery) => {
    setQuery(exampleQuery)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Recall</h1>
              <p className="text-sm text-gray-600">Your personal memory assistant</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-12">
        {/* Search Card */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-blue-600" />
              Ask Anything About Your Activities
            </CardTitle>
            <CardDescription>
              Query your personal activity data using natural language
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="What Italian restaurants did I visit?"
                  className="w-full min-h-[100px] p-4 pr-12 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                  disabled={loading}
                />
                <Search className="absolute right-4 top-4 w-5 h-5 text-gray-400" />
              </div>

              <Button type="submit" disabled={loading} className="w-full">
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                    Searching...
                  </>
                ) : (
                  'Search'
                )}
              </Button>
            </form>

            {/* Example Queries */}
            <div className="mt-6">
              <p className="text-sm text-gray-600 mb-3">Try these examples:</p>
              <div className="flex flex-wrap gap-2">
                {exampleQueries.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleExampleClick(example)}
                    className="text-sm px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-full text-gray-700 transition-colors"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Answer Card */}
        {answer && (
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">
                {answer.error ? 'Error' : 'Answer'}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {answer.error ? (
                <p className="text-red-600">{answer.message}</p>
              ) : (
                <div className="prose prose-sm max-w-none">
                  <div dangerouslySetInnerHTML={{ __html: answer.answer?.replace(/\n/g, '<br>') }} />

                  {answer.sources && answer.sources.length > 0 && (
                    <div className="mt-6 pt-4 border-t">
                      <p className="text-sm font-semibold text-gray-700 mb-2">Sources:</p>
                      <div className="space-y-2">
                        {answer.sources.map((source, idx) => (
                          <div key={idx} className="text-sm bg-gray-50 p-3 rounded">
                            <p className="text-gray-600">{source.text}</p>
                            {source.metadata && (
                              <p className="text-xs text-gray-500 mt-1">
                                {source.metadata.user_id && `User: ${source.metadata.user_id}`}
                                {source.metadata.timestamp && ` • ${new Date(source.metadata.timestamp).toLocaleDateString()}`}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Info Section */}
        {!answer && (
          <div className="text-center mt-12 text-gray-600">
            <Brain className="w-12 h-12 mx-auto mb-4 text-blue-400" />
            <p className="text-lg">Powered by hybrid RAG with vector search, BM25, and knowledge graphs</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-20 py-8 text-center text-sm text-gray-600 border-t">
        <p>Built with FastAPI, Qdrant, Groq, and FastEmbed</p>
      </footer>
    </div>
  )
}

export default App
