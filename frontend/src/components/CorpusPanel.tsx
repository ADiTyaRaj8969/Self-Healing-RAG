import { useRef, useState } from 'react'
import { clearCorpus, uploadDocuments } from '../api'
import type { Corpus, UploadResult } from '../types'

interface Props {
  corpus: Corpus | null
  accepts: string[]
  onCorpusChange: (c: Corpus) => void
  disabled: boolean
}

export default function CorpusPanel({ corpus, accepts, onCorpusChange, disabled }: Props) {
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [results, setResults] = useState<UploadResult[]>([])
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function handleFiles(fileList: FileList | null) {
    const files = Array.from(fileList ?? [])
    if (!files.length) return

    setBusy(true)
    setError(null)
    setResults([])
    try {
      const res = await uploadDocuments(files)
      setResults(res.results)
      onCorpusChange(res.corpus)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  async function handleClear() {
    setBusy(true)
    setError(null)
    try {
      const res = await clearCorpus()
      onCorpusChange(res.corpus)
      setResults([])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const count = corpus?.count ?? 0
  const sources = corpus?.sources ?? []
  const failures = results.filter((r) => !r.ok)

  return (
    <div className="corpus">
      <div
        className={'drop' + (dragging ? ' dragging' : '') + (busy ? ' busy' : '')}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (!disabled && !busy) handleFiles(e.dataTransfer.files)
        }}
        onClick={() => !disabled && !busy && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accepts.join(',')}
          multiple
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
        <span className="drop-text">
          {busy ? 'Extracting & embedding…' : dragging ? 'Release to add' : 'Add documents'}
        </span>
        <span className="drop-sub">
          drag or click · {accepts.join(' ')} · 20 MB
        </span>
      </div>

      {error && <p className="corpus-error">{error}</p>}

      {failures.map((r) => (
        <div key={r.filename} className="file-row bad">
          <span className="file-name">{r.filename}</span>
          <span className="file-dots" />
          <span className="file-note">{r.error}</span>
        </div>
      ))}

      {sources.length > 0 ? (
        <div className="corpus-detail">
          {sources.map((s) => (
            <div key={s.source} className="file-row">
              <span className="file-name">{s.source}</span>
              <span className="file-dots" />
              <span className="file-note">{s.chunks} chunks</span>
            </div>
          ))}
          <div className="corpus-foot">
            <span className="corpus-total">
              {count} chunks · {sources.length} file{sources.length === 1 ? '' : 's'}
            </span>
            <button className="clear-btn" onClick={handleClear} disabled={busy || disabled}>
              clear all
            </button>
          </div>
        </div>
      ) : (
        <p className="corpus-empty">No documents yet. Add one to build the corpus.</p>
      )}
    </div>
  )
}
