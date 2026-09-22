

import React, { useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = import.meta.env.VITE_API_URL || '';

const examples = [
  'Why was Barclays fined in 2025?',
  'What are the most common issues banks were fined for?',
  'Compare Barclays fines in 2024 and 2025.',
  'Which firm received the largest fine in 2025?',
];

function formatDuration(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null;
  }

  return `${Math.round(value)} ms`;
}

function App() {
  const [q, setQ] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  // Prevent duplicate requests while React state is updating.
  const requestInFlight = useRef(false);

  async function ask(question = q) {
    const trimmed = question.trim();

    if (!trimmed || requestInFlight.current) {
      return;
    }

    requestInFlight.current = true;
    setLoading(true);

    const assistantId = `assistant-${Date.now()}-${Math.random()}`;

    setMessages((m) => [
      ...m,
      {
        role: 'user',
        text: trimmed,
      },
      {
        id: assistantId,
        role: 'assistant',
        text: '',
        sources: [],
        streaming: true,
      },
    ]);

    setQ('');

    const clientStarted = performance.now();

    try {
      const response = await fetch(`${API}/api/ask/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: trimmed,
        }),
      });

      if (!response.ok) {
        let detail = 'Request failed';
        try {
          const data = await response.json();
          detail = data.detail || detail;
        } catch {
          // Keep the generic error when the server didn't return JSON.
        }
        throw new Error(detail);
      }

      if (!response.body) {
        throw new Error('The server did not provide a response stream.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let completed = false;

      function applyEvent(event) {
        if (event.type === 'token') {
          setMessages((current) => current.map((message) => (
            message.id === assistantId
              ? { ...message, text: message.text + event.text }
              : message
          )));
          return;
        }

        if (event.type === 'complete') {
          const data = event.data;
          completed = true;
          setMessages((current) => current.map((message) => (
            message.id === assistantId
              ? {
                  ...message,
                  text: data.answer,
                  sources: data.sources || [],
                  request_duration_ms: data.request_duration_ms,
                  response_duration_ms: data.response_duration_ms,
                  client_duration_ms: performance.now() - clientStarted,
                  streaming: false,
                }
              : message
          )));
          return;
        }

        if (event.type === 'error') {
          throw new Error(event.detail || 'The answer stream ended unexpectedly.');
        }
      }

      function consumeLines(final = false) {
        const lines = buffer.split('\n');
        buffer = final ? '' : lines.pop();
        for (const line of lines) {
          if (line.trim()) {
            applyEvent(JSON.parse(line));
          }
        }
        if (final && buffer.trim()) {
          applyEvent(JSON.parse(buffer));
          buffer = '';
        }
      }

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        consumeLines(done);
        if (done) break;
      }

      if (!completed) {
        throw new Error('The answer stream ended before completion.');
      }
    } catch (error) {
      console.error('API request failed:', error);

      setMessages((current) => current.map((message) => (
        message.id === assistantId
          ? {
              ...message,
              text: 'Sorry, I could not answer that request. Please check that the backend is running and indexed.',
              sources: [],
              streaming: false,
            }
          : message
      )));
    } finally {
      requestInFlight.current = false;
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <div>
          <div className="eyebrow">FCA ENFORCEMENT RESEARCH</div>

          <h1>Fines RAG Assistant</h1>

          <p>
            Ask questions about FCA fines and Final Notices from 2024–2026.
          </p>
        </div>

        <div className="badge">RAG · Sources included</div>
      </header>

      <section className="examples">
        {examples.map((example) => (
          <button
            key={example}
            type="button"
            disabled={loading}
            onClick={() => ask(example)}
          >
            {example}
          </button>
        ))}
      </section>

      <section className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <h2>What would you like to know?</h2>
            <p>
              Try a company, year, reason, comparison or trend question.
            </p>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`msg ${message.role}`}
          >
            <div className="bubble">{message.text}</div>

            {(
              typeof message.request_duration_ms === 'number' ||
              typeof message.response_duration_ms === 'number' ||
              typeof message.client_duration_ms === 'number'
            ) && (
              <div className="meta">
                {typeof message.request_duration_ms === 'number' && (
                  <span>
                    Backend: {formatDuration(message.request_duration_ms)}
                  </span>
                )}

                {typeof message.response_duration_ms === 'number' && (
                  <span>
                    Generation: {formatDuration(message.response_duration_ms)}
                  </span>
                )}

                {typeof message.client_duration_ms === 'number' && (
                  <span>
                    Total: {formatDuration(message.client_duration_ms)}
                  </span>
                )}
              </div>
            )}

            {message.sources?.length > 0 && (
              <div className="sources">
                <b>Sources</b>

                {message.sources.map((source, sourceIndex) => (
                  <a
                    key={source.source_id || sourceIndex}
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {source.citation ? `[${source.citation}] ` : ''}
                    {source.firm || source.title} · {source.year} · p.
                    {source.page || '—'}
                  </a>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="msg assistant">
            <div className="bubble">
              {messages.some((message) => message.role === 'assistant' && message.streaming && message.text)
                ? 'Generating answer…'
                : 'Searching FCA documents…'}
            </div>
          </div>
        )}
      </section>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          ask();
        }}
      >
        <input
          value={q}
          onChange={(event) => setQ(event.target.value)}
          placeholder="Ask about an FCA fine…"
          disabled={loading}
        />

        <button
          type="submit"
          disabled={loading || !q.trim()}
        >
          {loading ? 'Searching…' : 'Ask'}
        </button>
      </form>

      <footer>
        Grounded only in the indexed FCA documents.{' '}
        <a
          href="https://www.fca.org.uk/news/news-stories/2024-fines"
          target="_blank"
          rel="noopener noreferrer"
        >
          FCA 2024
        </a>{' '}
        ·{' '}
        <a
          href="https://www.fca.org.uk/news/news-stories/2025-fines"
          target="_blank"
          rel="noopener noreferrer"
        >
          FCA 2025
        </a>{' '}
        ·{' '}
        <a
          href="https://www.fca.org.uk/news/news-stories/2026-fines"
          target="_blank"
          rel="noopener noreferrer"
        >
          FCA 2026
        </a>
      </footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);