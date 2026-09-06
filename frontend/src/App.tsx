import { useEffect, useRef, useState } from 'react'
import { clearCorpus, fetchHealth, streamAsk } from './api'
import AttemptCard from './components/AttemptCard'
import CorpusPanel from './components/CorpusPanel'
import FinalAnswer from './components/FinalAnswer'
import GraphDiagram from './components/GraphDiagram'
import Logo from './components/Logo'
import type { Attempt, Corpus, Health, Stage, Verdict } from './types'

export default function App() {
  const [question, setQuestion] = useState('')
  const [maxAttempts, setMaxAttempts] = useState(3)
  const [attempts, setAttempts] = useState<Attempt[]>([])
  const [stage, setStage] = useState<Stage>('idle')
  const [final, setFinal] = useState<{ status: 'answered' | 'refused'; answer: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const [corpus, setCorpus] = useState<Corpus | null>(null)

  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const bootedRef = useRef(false)

  // The corpus is per-session: every page load starts empty, so a reload never
  // leaves documents from a previous session lying around. (StrictMode runs
  // effects twice in dev — the ref keeps this to a single round trip.)
  useEffect(() => {
    if (bootedRef.current) return
    bootedRef.current = true

    const loadHealth = () =>
      fetchHealth()
        .then((h) => {
          setHealth(h)
          setCorpus(h.corpus)
        })
        .catch(() => setHealth(null))

    clearCorpus().catch(() => {}).finally(loadHealth)
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [attempts.length, final])

  const retries = Math.max(0, attempts.length - 1)
  const lastVerdict: Verdict | null = attempts[attempts.length - 1]?.verdict ?? null
  const hasDocs = (corpus?.count ?? 0) > 0

  async function run(q: string) {
    const trimmed = q.trim()
    if (!trimmed || running) return

    setAttempts([])
    setFinal(null)
    setError(null)
    setRunning(true)
    setStage('retrieve')

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamAsk(
        trimmed,
        maxAttempts,
        (ev) => {
          switch (ev.type) {
            case 'retrieve':
              setStage('retrieve')
              setAttempts((prev) => [
                ...prev,
                { n: prev.length + 1, query: ev.query, chunks: ev.chunks },
              ])
              break
            case 'generate':
              setStage('generate')
              setAttempts((prev) =>
                prev.map((a, i) => (i === prev.length - 1 ? { ...a, answer: ev.answer } : a)),
              )
              break
            case 'critique':
              setStage('critique')
              setAttempts((prev) =>
                prev.map((a, i) =>
                  i === prev.length - 1
                    ? {
                        ...a,
                        verdict: ev.verdict,
                        reasoning: ev.reasoning,
                        reformulatedQuery: ev.reformulatedQuery,
                      }
                    : a,
                ),
              )
              break
            case 'final':
              setFinal({ status: ev.status, answer: ev.answer })
              setStage('done')
              break
            case 'error':
              setError(ev.message)
              break
          }
        },
        controller.signal,
      )
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError((e as Error).message)
    } finally {
      setRunning(false)
      abortRef.current = null
      setStage((s) => (s === 'done' ? 'done' : 'idle'))
    }
  }

  const stageLabel = running ? stage : final ? final.status : 'idle'

  return (
    <div className="page">
      <header className="topbar">
        <div className="topbar-brand">
          <Logo />
          <span className="wordmark">
            Self&#8209;Healing<span className="wordmark-slash">/</span>RAG
          </span>
        </div>
        {health ? (
          <div className="topbar-meta">
            <span>
              <em>provider</em>
              {health.provider}
            </span>
            <span>
              <em>model</em>
              {health.generationModel}
            </span>
            <span>
              <em>top-k</em>
              {health.topK}
            </span>
            <span className={'state ' + (running ? 'on' : '')}>
              <i />
              {stageLabel}
            </span>
          </div>
        ) : (
          <div className="topbar-meta off">
            <span>backend offline</span>
          </div>
        )}
      </header>

      <div className="grid">
        <aside className="rail">
          <div className="lede">
            <h1 className="title">
              Self-Healing <em>RAG</em>
            </h1>
            <p className="standfirst">
              A retrieval pipeline that judges its own answer, reformulates the query when
              the answer isn't grounded, and declines rather than inventing one.
            </p>
          </div>

          <div className="block">
            <h2 className="block-h">
              <span className="block-n">01</span> Corpus
            </h2>
            <CorpusPanel
              corpus={corpus}
              accepts={health?.accepts ?? ['.pdf', '.txt', '.md']}
              onCorpusChange={setCorpus}
              disabled={running}
            />
          </div>

          <div className="block">
            <h2 className="block-h">
              <span className="block-n">02</span> Query
            </h2>
            <form
              className="ask"
              onSubmit={(e) => {
                e.preventDefault()
                run(question)
              }}
            >
              <input
                className="ask-input"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={hasDocs ? 'Ask your documents…' : 'Add a document first…'}
                disabled={running || !hasDocs}
              />
              <div className="ask-controls">
                <label className="passes" title="Maximum retrieve/generate/critique cycles">
                  <span>max passes</span>
                  <input
                    type="number"
                    min={1}
                    max={6}
                    value={maxAttempts}
                    disabled={running}
                    onChange={(e) => setMaxAttempts(Number(e.target.value))}
                  />
                </label>
                {running ? (
                  <button
                    type="button"
                    className="btn btn-stop"
                    onClick={() => abortRef.current?.abort()}
                  >
                    Stop
                  </button>
                ) : (
                  <button type="submit" className="btn" disabled={!question.trim() || !hasDocs}>
                    Run
                  </button>
                )}
              </div>
            </form>
          </div>
        </aside>

        <main className="main">
          <GraphDiagram
            stage={stage}
            retries={retries}
            finalStatus={final?.status ?? null}
            lastVerdict={lastVerdict}
            running={running}
          />

          {error && <p className="error">{error}</p>}

          <div className="block">
            <h2 className="block-h">
              <span className="block-n">03</span> Trace
              {attempts.length > 0 && (
                <span className="block-note">
                  {attempts.length} pass{attempts.length === 1 ? '' : 'es'}
                  {retries > 0 && ` · ${retries} retr${retries === 1 ? 'y' : 'ies'}`}
                </span>
              )}
            </h2>

            {attempts.length > 0 ? (
              <div className="trace">
                {attempts.map((a, i) => (
                  <AttemptCard
                    key={a.n}
                    attempt={a}
                    isLast={i === attempts.length - 1}
                    running={running}
                  />
                ))}
              </div>
            ) : (
              <p className="empty">
                {hasDocs
                  ? 'No passes recorded. Ask a question to trace the loop — anything your documents don’t cover is refused rather than guessed.'
                  : 'Add a document to get started. Everything the pipeline answers comes from the files you supply.'}
              </p>
            )}
          </div>

          {final && (
            <div className="block">
              <h2 className="block-h">
                <span className="block-n">04</span> Result
              </h2>
              <FinalAnswer
                status={final.status}
                answer={final.answer}
                attempts={attempts.length}
              />
            </div>
          )}

          <div ref={bottomRef} />
        </main>
      </div>
    </div>
  )
}
