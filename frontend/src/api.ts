import type { Corpus, Health, StreamEvent, UploadResult } from './types'

async function errorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body = await res.json()
    if (typeof body?.detail === 'string') return body.detail
  } catch {
    /* non-JSON error body */
  }
  return fallback
}

/** Streams the pipeline's server-sent events, invoking `onEvent` per node completion. */
export async function streamAsk(
  question: string,
  maxAttempts: number,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, max_attempts: maxAttempts }),
    signal,
  })

  if (!res.ok || !res.body) {
    throw new Error(
      await errorMessage(res, `Backend returned ${res.status}. Is the server running on :8000?`),
    )
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    // SSE frames are separated by a blank line; the trailing partial stays buffered.
    const frames = buffer.split('\n\n')
    buffer = frames.pop() ?? ''

    for (const frame of frames) {
      const line = frame.split('\n').find((l) => l.startsWith('data: '))
      if (line) onEvent(JSON.parse(line.slice(6)) as StreamEvent)
    }
  }
}

export async function fetchHealth(): Promise<Health> {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error(`Health check failed (${res.status})`)
  return res.json()
}

export async function uploadDocuments(
  files: File[],
): Promise<{ results: UploadResult[]; corpus: Corpus }> {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))

  const res = await fetch('/api/upload', { method: 'POST', body: form })
  if (!res.ok) throw new Error(await errorMessage(res, `Upload failed (${res.status})`))
  return res.json()
}

export async function clearCorpus(): Promise<{ corpus: Corpus }> {
  const res = await fetch('/api/corpus', { method: 'DELETE' })
  if (!res.ok) throw new Error(await errorMessage(res, `Clear failed (${res.status})`))
  return res.json()
}
