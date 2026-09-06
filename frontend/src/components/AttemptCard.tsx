import type { Attempt, Verdict } from '../types'
import ChunkList from './ChunkList'

const VERDICT_LABEL: Record<Verdict, string> = {
  grounded: 'grounded',
  hallucinated: 'hallucinated',
  insufficient_information: 'insufficient',
}

export default function AttemptCard({
  attempt,
  isLast,
  running,
}: {
  attempt: Attempt
  isLast: boolean
  running: boolean
}) {
  const pending = isLast && running && !attempt.verdict

  return (
    <article className={'pass' + (attempt.verdict ? ` v-${attempt.verdict}` : '')}>
      {/* verdict lives in the margin, like an annotation on a manuscript */}
      <div className="pass-margin">
        <span className="pass-n">{String(attempt.n).padStart(2, '0')}</span>
        {attempt.verdict ? (
          <span className="verdict">{VERDICT_LABEL[attempt.verdict]}</span>
        ) : (
          <span className="verdict pending">{pending ? 'running' : 'queued'}</span>
        )}
      </div>

      <div className="pass-body">
        <div className="row">
          <span className="row-key">query</span>
          <code className="row-val mono">{attempt.query}</code>
        </div>

        <ChunkList chunks={attempt.chunks} />

        {attempt.answer ? (
          <div className="row">
            <span className="row-key">answer</span>
            <p className="row-val">{attempt.answer}</p>
          </div>
        ) : (
          pending && (
            <div className="row">
              <span className="row-key">answer</span>
              <div className="row-val">
                <span className="skeleton" />
              </div>
            </div>
          )
        )}

        {attempt.reasoning && (
          <div className="row">
            <span className="row-key">critic</span>
            <p className="row-val critic">{attempt.reasoning}</p>
          </div>
        )}

        {attempt.reformulatedQuery && (
          <div className="row reform">
            <span className="row-key">retry</span>
            <code className="row-val mono retry-val">↻ {attempt.reformulatedQuery}</code>
          </div>
        )}
      </div>
    </article>
  )
}
