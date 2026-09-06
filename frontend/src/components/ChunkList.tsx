import { useState } from 'react'
import type { Chunk } from '../types'

/** Collapsed by default — retrieved context is supporting evidence, not the story. */
export default function ChunkList({ chunks }: { chunks: Chunk[] }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="row">
      <span className="row-key">context</span>
      <div className="row-val">
        <button className="ctx-toggle" onClick={() => setOpen(!open)}>
          {chunks.length} chunk{chunks.length === 1 ? '' : 's'} retrieved
          <span className={'caret' + (open ? ' open' : '')}>›</span>
        </button>

        {open && (
          <div className="ctx-body">
            {chunks.map((c, i) => (
              <figure key={i} className="ctx-chunk">
                <figcaption className="ctx-src">
                  <span className="ctx-i">[{i + 1}]</span>
                  {c.source}
                </figcaption>
                <pre className="ctx-text">{c.content}</pre>
              </figure>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
