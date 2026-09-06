export default function FinalAnswer({
  status,
  answer,
  attempts,
}: {
  status: 'answered' | 'refused'
  answer: string
  attempts: number
}) {
  return (
    <div className={`result r-${status}`}>
      <p className="result-text">{answer}</p>
      <div className="result-foot">
        <span className="result-stamp">{status}</span>
        <span className="result-meta">
          resolved in {attempts} pass{attempts === 1 ? '' : 'es'}
        </span>
      </div>
    </div>
  )
}
