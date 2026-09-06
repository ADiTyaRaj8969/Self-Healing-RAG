export type Verdict = 'grounded' | 'hallucinated' | 'insufficient_information'

export type Stage = 'idle' | 'retrieve' | 'generate' | 'critique' | 'done'

export interface Chunk {
  source: string
  content: string
}

export interface Attempt {
  n: number
  query: string
  chunks: Chunk[]
  answer?: string
  verdict?: Verdict
  reasoning?: string
  reformulatedQuery?: string | null
}

export interface CorpusSource {
  source: string
  chunks: number
}

export interface Corpus {
  count: number
  sources: CorpusSource[]
}

export interface Health {
  provider: string
  generationModel: string
  criticModel: string
  embeddingModel: string
  topK: number
  maxAttempts: number
  accepts: string[]
  corpus: Corpus
}

export interface UploadResult {
  filename: string
  ok: boolean
  chunks?: number
  error?: string
}

export type StreamEvent =
  | { type: 'start'; question: string; maxAttempts: number }
  | { type: 'retrieve'; query: string; chunks: Chunk[] }
  | { type: 'generate'; answer: string }
  | {
      type: 'critique'
      attempt: number
      verdict: Verdict
      reasoning: string
      reformulatedQuery: string | null
    }
  | { type: 'final'; status: 'answered' | 'refused'; answer: string }
  | { type: 'error'; message: string }
  | { type: 'done' }
