import type { Stage, Verdict } from '../types'

interface Props {
  stage: Stage
  retries: number
  finalStatus: 'answered' | 'refused' | null
  lastVerdict: Verdict | null
  running: boolean
}

const BOX_W = 132
const BOX_H = 52

const NODES = {
  retrieve: { x: 26, y: 74, label: 'RETRIEVE', sub: 'top-k search' },
  generate: { x: 206, y: 74, label: 'GENERATE', sub: 'draft answer' },
  critique: { x: 386, y: 74, label: 'CRITIQUE', sub: 'groundedness' },
  finalize: { x: 596, y: 22, label: 'FINALIZE', sub: 'accept' },
  refuse: { x: 596, y: 126, label: 'REFUSE', sub: 'decline' },
} as const

type NodeKey = keyof typeof NODES

// Which nodes have already been visited by the time we reach a given stage.
const VISITED: Record<Stage, NodeKey[]> = {
  idle: [],
  retrieve: [],
  generate: ['retrieve'],
  critique: ['retrieve', 'generate'],
  done: ['retrieve', 'generate', 'critique'],
}

/** Arrowheads are separate markers per state so each head matches its edge's colour.
 *  Three details matter for these to look clean:
 *   - markerUnits="userSpaceOnUse" keeps them one size regardless of stroke-width
 *   - refX=0 anchors the *base* at the path end, so the head extends forward into the
 *     node and the stroke never runs underneath it
 *   - the fills are opaque (see .ah-* rules); a translucent head lets the edge show
 *     through and reads as a doubled, muddy shape
 *  Proportions are 10 long x 6.4 wide — noticeably sharper than an equilateral head. */
function Arrowhead({ id, cls }: { id: string; cls: string }) {
  return (
    <marker
      id={id}
      viewBox="0 0 10 6.4"
      refX="0"
      refY="3.2"
      markerWidth="10"
      markerHeight="6.4"
      markerUnits="userSpaceOnUse"
      orient="auto"
    >
      <path d="M 0 0 L 10 3.2 L 0 6.4 Z" className={cls} />
    </marker>
  )
}

/** How far each edge stops short of its target node, leaving room for the head. */
const HEAD = 10

export default function GraphDiagram({
  stage,
  retries,
  finalStatus,
  lastVerdict,
  running,
}: Props) {
  const visited = VISITED[stage]

  function nodeClass(key: NodeKey): string {
    if (key === 'finalize' || key === 'refuse') {
      const active =
        (key === 'finalize' && finalStatus === 'answered') ||
        (key === 'refuse' && finalStatus === 'refused')
      return active ? `node node-terminal active ${key}` : 'node node-terminal'
    }
    if (stage === key && running) return 'node active'
    if (visited.includes(key)) return 'node visited'
    return 'node'
  }

  // The critique node carries the verdict colour once a verdict exists.
  const critiqueTone = lastVerdict && stage !== 'critique' ? ` tone-${lastVerdict}` : ''

  const edgeClass = (active: boolean, done: boolean) =>
    'edge' + (active ? ' edge-active' : '') + (done ? ' edge-done' : '')

  const marker = (active: boolean, done: boolean) =>
    active ? 'url(#ah-active)' : done ? 'url(#ah-done)' : 'url(#ah-idle)'

  const genActive = stage === 'generate' && running
  const critActive = stage === 'critique' && running
  const toFinalize = finalStatus === 'answered'
  const toRefuse = finalStatus === 'refused'
  const retryLit = retries > 0

  return (
    <figure className="figure">
      <div className="screen">
        <div className="screen-head">
          <span className="screen-title">langgraph · state</span>
          <span className="screen-hint">
            {running ? 'running' : finalStatus ? `halted — ${finalStatus}` : 'idle'}
          </span>
        </div>

        <svg className="graph-svg" viewBox="0 0 760 232" role="img" aria-label="Pipeline graph">
          <defs>
            <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <Arrowhead id="ah-idle" cls="ah ah-idle" />
            <Arrowhead id="ah-done" cls="ah ah-done" />
            <Arrowhead id="ah-active" cls="ah ah-active" />
            <Arrowhead id="ah-retry" cls="ah ah-retry" />
            <Arrowhead id="ah-retry-on" cls="ah ah-retry-on" />
          </defs>

          {/* retrieve -> generate */}
          <path
            d={`M ${NODES.retrieve.x + BOX_W} 100 L ${NODES.generate.x - HEAD} 100`}
            className={edgeClass(genActive, visited.includes('generate'))}
            markerEnd={marker(genActive, visited.includes('generate'))}
          />
          {/* generate -> critique */}
          <path
            d={`M ${NODES.generate.x + BOX_W} 100 L ${NODES.critique.x - HEAD} 100`}
            className={edgeClass(critActive, visited.includes('critique'))}
            markerEnd={marker(critActive, visited.includes('critique'))}
          />
          {/* critique -> finalize */}
          <path
            d={`M ${NODES.critique.x + BOX_W} 96 C 552 96, 562 48, ${NODES.finalize.x - HEAD} 48`}
            className={edgeClass(false, toFinalize)}
            markerEnd={marker(false, toFinalize)}
          />
          {/* critique -> refuse */}
          <path
            d={`M ${NODES.critique.x + BOX_W} 104 C 552 104, 562 152, ${NODES.refuse.x - HEAD} 152`}
            className={edgeClass(false, toRefuse)}
            markerEnd={marker(false, toRefuse)}
          />

          {/* the retry cycle — the edge that makes this a graph and not a chain */}
          <path
            d={`M 452 ${NODES.critique.y + BOX_H} L 452 186 Q 452 198 440 198 L 104 198 Q 92 198 92 186 L 92 ${NODES.retrieve.y + BOX_H + HEAD}`}
            className={
              'edge edge-retry' +
              (retryLit ? ' edge-retry-fired' : '') +
              (stage === 'retrieve' && running && retryLit ? ' edge-active-retry' : '')
            }
            markerEnd={retryLit ? 'url(#ah-retry-on)' : 'url(#ah-retry)'}
          />
          <g className={'retry-badge' + (retryLit ? ' on' : '')}>
            <rect x="238" y="185" width="82" height="26" rx="13" />
            <text x="279" y="202">
              retry {retryLit ? `×${retries}` : ''}
            </text>
          </g>

          {(Object.keys(NODES) as NodeKey[]).map((key) => {
            const n = NODES[key]
            const cls = nodeClass(key) + (key === 'critique' ? critiqueTone : '')
            return (
              <g key={key} className={cls}>
                <rect x={n.x} y={n.y} width={BOX_W} height={BOX_H} rx="11" className="node-box" />
                <text x={n.x + BOX_W / 2} y={n.y + 22} className="node-label">
                  {n.label}
                </text>
                <text x={n.x + BOX_W / 2} y={n.y + 38} className="node-sub">
                  {n.sub}
                </text>
              </g>
            )
          })}
        </svg>

        <div className="graph-legend">
          <span><i className="dot dot-active" /> active</span>
          <span><i className="dot dot-visited" /> visited</span>
          <span><i className="dot dot-retry" /> retry edge</span>
        </div>
      </div>

      <figcaption className="figure-caption">
        <b>Fig. 1</b> — Pipeline state. Nodes illuminate as each LangGraph node executes;
        the amber edge is the cycle back to retrieval, which fires when the critic rejects
        an answer. A linear chain cannot express that edge.
      </figcaption>
    </figure>
  )
}
