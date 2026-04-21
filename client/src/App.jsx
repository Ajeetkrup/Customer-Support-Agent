import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_BASE = ''

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function TypingDots() {
  return (
    <div className="typing-dots" aria-label="Agent is thinking">
      <span></span>
      <span></span>
      <span></span>
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`message-row ${isUser ? 'user-row' : 'agent-row'}`}>
      {!isUser && (
        <div className="avatar agent-avatar" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" fill="url(#agentGrad)" />
            <path d="M8 12h8M12 8v8" stroke="white" strokeWidth="2" strokeLinecap="round" />
            <defs>
              <linearGradient id="agentGrad" x1="2" y1="2" x2="22" y2="22">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#a855f7" />
              </linearGradient>
            </defs>
          </svg>
        </div>
      )}
      <div className={`message-bubble ${isUser ? 'user-bubble' : 'agent-bubble'}`}>
        {msg.loading ? (
          <TypingDots />
        ) : (
          <p className="message-text">{msg.content}</p>
        )}
        {!msg.loading && (
          <span className={`message-time ${isUser ? 'message-time-user-time' : ''}`}>{formatTime(msg.timestamp)}</span>
        )}
      </div>
      {isUser && (
        <div className="avatar user-avatar" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="8" r="4" fill="currentColor" />
            <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" fill="currentColor" />
          </svg>
        </div>
      )}
    </div>
  )
}

const SUGGESTED_QUERIES = [
  'What is the return window for electronics?',
  'How long does a refund take to process?',
  'Can I return a product I have already used?',
  'What items are non-returnable?',
]

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'agent',
      content: 'Hello! I\'m your Customer Support assistant. Ask me anything about returns, refunds, warranties, or any other customer support topics.',
      timestamp: new Date(),
      loading: false,
    },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)
  const textareaRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = async (queryText) => {
    const query = (queryText || input).trim()
    if (!query || isLoading) return

    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: query,
      timestamp: new Date(),
      loading: false,
    }

    const loadingMsg = {
      id: `loading-${Date.now()}`,
      role: 'agent',
      content: '',
      timestamp: new Date(),
      loading: true,
    }

    setMessages(prev => [...prev, userMsg, loadingMsg])
    setInput('')
    setIsLoading(true)

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    try {
      abortRef.current = new AbortController()
      const params = new URLSearchParams({ query })
      const res = await fetch(`${API_BASE}/api/retrieve-knowledge-base?${params}`, {
        method: 'POST',
        signal: abortRef.current.signal,
      })

      if (!res.ok) {
        throw new Error(`API error: ${res.status} ${res.statusText}`)
      }

      const data = await res.json()
      const answer = data.result || data.message || 'Sorry, I couldn\'t find a relevant answer.'

      setMessages(prev => prev.map(m =>
        m.loading ? { ...m, content: answer, loading: false, timestamp: new Date() } : m
      ))
    } catch (err) {
      if (err.name === 'AbortError') return
      setMessages(prev => prev.map(m =>
        m.loading
          ? {
            ...m,
            content: '⚠️ Something went wrong while fetching the answer. Please try again.',
            loading: false,
            timestamp: new Date(),
            error: true,
          }
          : m
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleTextareaChange = (e) => {
    setInput(e.target.value)
    const ta = textareaRef.current
    if (ta) {
      ta.style.height = 'auto'
      ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`
    }
  }

  const handleSuggestion = (q) => {
    if (!isLoading) handleSubmit(q)
  }

  return (
    <div className="chat-app">
      {/* Header */}
      <header className="chat-header">
        <div className="header-brand">
          <div className="header-logo" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
              <rect width="32" height="32" rx="10" fill="url(#logoGrad)" />
              <path d="M10 16h12M16 10v12" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
              <circle cx="16" cy="16" r="4" stroke="white" strokeWidth="2" />
              <defs>
                <linearGradient id="logoGrad" x1="0" y1="0" x2="32" y2="32">
                  <stop offset="0%" stopColor="#6366f1" />
                  <stop offset="100%" stopColor="#a855f7" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div className="header-info">
            <h1 className="header-title">Customer Support Agent</h1>
            <span className="header-subtitle">
              <span className="status-dot" aria-hidden="true"></span>
              AI Knowledge Assistant
            </span>
          </div>
        </div>
        <div className="header-badge">RAG Powered</div>
      </header>

      {/* Messages */}
      <main className="messages-area" role="log" aria-live="polite" aria-label="Chat messages">
        {messages.map(msg => (
          <Message key={msg.id} msg={msg} />
        ))}
        <div ref={messagesEndRef} />
      </main>

      {/* Suggestions — only show when idle and no user messages yet */}
      {messages.length === 1 && !isLoading && (
        <div className="suggestions-area" role="list" aria-label="Suggested questions">
          <p className="suggestions-label">Try asking:</p>
          <div className="suggestions-grid">
            {SUGGESTED_QUERIES.map((q, i) => (
              <button
                key={i}
                className="suggestion-chip"
                onClick={() => handleSuggestion(q)}
                role="listitem"
                aria-label={`Ask: ${q}`}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <footer className="input-area">
        <form
          className="input-form"
          onSubmit={(e) => { e.preventDefault(); handleSubmit() }}
          aria-label="Send a message"
        >
          <div className={`input-wrapper ${isLoading ? 'loading' : ''}`}>
            <textarea
              ref={textareaRef}
              id="chat-input"
              className="chat-textarea"
              value={input}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask about returns, refunds, shipping, warranties…"
              rows={1}
              disabled={isLoading}
              aria-label="Your question"
              aria-describedby="input-hint"
            />
            <button
              id="send-btn"
              type="submit"
              className={`send-btn ${isLoading ? 'send-btn-loading' : ''}`}
              disabled={isLoading || !input.trim()}
              aria-label={isLoading ? 'Waiting for response…' : 'Send message'}
            >
              <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M22 2L15 22l-4-9-9-4 20-7z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
          <p id="input-hint" className="input-hint">
            {isLoading
              ? '⏳ Searching knowledge base…'
              : 'Press Enter to send · Shift+Enter for new line'}
          </p>
        </form>
      </footer>
    </div>
  )
}
