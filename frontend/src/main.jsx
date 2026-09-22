

import React, { useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'https://combinations-landscapes-featuring-instructional.trycloudflare.com';

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

    setMessages((m) => [
      ...m,
      {
        role: 'user',
        text: trimmed,
      },
    ]);

    setQ('');

    const clientStarted = performance.now();

    try {
      const response = await fetch(`${API}/api/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: trimmed,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Request failed');
      }

      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text: data.answer,
          sources: data.sources || [],
          request_duration_ms: data.request_duration_ms,
          response_duration_ms: data.response_duration_ms,
          client_duration_ms: performance.now() - clientStarted,
        },
      ]);
    } catch (error) {
      console.error('API request failed:', error);

      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          text:
            'Sorry, I could not answer that request. Please check that the backend is running and indexed.',
        },
      ]);
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
                    key={sourceIndex}
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
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
              Searching FCA documents…
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