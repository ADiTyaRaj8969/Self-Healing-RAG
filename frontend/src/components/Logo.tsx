/** The pipeline shrunk to a mark: three nodes in sequence with the retry edge
 *  looping back. Drawn in ink, so it sits on the paper like a printed diagram. */
export default function Logo() {
  return (
    <svg className="logo" viewBox="0 0 34 21" role="img" aria-label="Self-Healing RAG">
      <path d="M 8 8 L 14 8" className="logo-edge" />
      <path d="M 20 8 L 26 8" className="logo-edge" />

      <path
        d="M 29 11 L 29 16 Q 29 18 27 18 L 7 18 Q 5 18 5 16 L 5 13.8"
        className="logo-retry"
      />
      <polygon points="5,10.4 3.4,13.9 6.6,13.9" className="logo-arrow" />

      <rect x="2" y="5" width="6" height="6" className="logo-node" />
      <rect x="14" y="5" width="6" height="6" className="logo-node mid" />
      <rect x="26" y="5" width="6" height="6" className="logo-node" />
    </svg>
  )
}
