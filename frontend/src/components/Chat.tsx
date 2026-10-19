import { useEffect, useRef, useState } from 'react'
import { api, streamMessage, type Citation, type Message, type TraceStage } from '../api'

function CitationChips({ citations, onOpenPdf }: {
  citations: Citation[]
  onOpenPdf: (paperId: number, page: number) => void
}) {
  if (!citations.length) return null
  return (
    <div className="citations">
      {citations.map((c) => (
        <button key={c.marker} className="citation-chip" title={c.snippet}
                onClick={() => onOpenPdf(c.paper_id, c.page_start)}>
          [{c.marker}] {c.title.slice(0, 40)} · p.{c.page_start}
        </button>
      ))}
    </div>
  )
}

function Trace({ messageId }: { messageId: number }) {
  const [stages, setStages] = useState<TraceStage[] | null>(null)
  return (
    <details className="trace" onToggle={(e) => {
      if ((e.target as HTMLDetailsElement).open && !stages) {
        api.trace(messageId).then(setStages).catch(() => setStages([]))
      }
    }}>
      <summary>trace</summary>
      {stages?.map((s, i) => (
        <div key={i} className="trace-stage">
          <b>{s.stage}</b> {s.duration_ms}ms
          <pre>{JSON.stringify(s.payload, null, 1)}</pre>
        </div>
      ))}
    </details>
  )
}

export function Chat({ chatId, locked, onOpenPdf, onTitleChanged }: {
  chatId: number
  locked: boolean
  onOpenPdf: (paperId: number, page: number) => void
  onTitleChanged: () => void
}) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [draft, setDraft] = useState('')
  const [draftCitations, setDraftCitations] = useState<Citation[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setMessages([])
    api.chat(chatId).then((c) => setMessages(c.messages))
  }, [chatId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, draft])

  const send = async () => {
    const content = input.trim()
    if (!content || streaming) return
    setInput('')
    setStreaming(true)
    setDraft('')
    setDraftCitations([])
    setMessages((m) => [...m, { id: -1, role: 'user', content, citations: [] }])
    let text = ''
    let citations: Citation[] = []
    try {
      await streamMessage(chatId, content, (ev) => {
        if (ev.type === 'token') {
          text += ev.content
          setDraft(text)
        } else if (ev.type === 'citations') {
          citations = ev.citations
          setDraftCitations(citations)
        } else if (ev.type === 'done') {
          setMessages((m) => [...m, { id: ev.message_id, role: 'assistant', content: text, citations }])
          setDraft('')
          onTitleChanged()
        } else if (ev.type === 'error') {
          throw new Error(ev.message)
        }
      })
    } catch (e) {
      setMessages((m) => [...m, { id: -2, role: 'assistant', content: `Error: ${(e as Error).message}`, citations: [] }])
      setDraft('')
    } finally {
      setStreaming(false)
    }
  }

  return (
    <section className="chat">
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="msg-content">{m.content}</div>
            <CitationChips citations={m.citations} onOpenPdf={onOpenPdf} />
            {m.role === 'assistant' && m.id > 0 && <Trace messageId={m.id} />}
          </div>
        ))}
        {draft && (
          <div className="msg assistant">
            <div className="msg-content">{draft}<span className="cursor">▋</span></div>
            <CitationChips citations={draftCitations} onOpenPdf={onOpenPdf} />
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      {locked && (
        <div className="lock-banner">
          Analyzing papers… chat unlocks when every paper is ready.
        </div>
      )}
      <form className="composer" onSubmit={(e) => { e.preventDefault(); send() }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={locked ? 'Waiting for papers to finish analyzing…' : 'Ask about your papers…'}
          disabled={streaming || locked}
          autoFocus
        />
        <button disabled={streaming || locked || !input.trim()}>{streaming ? '…' : 'Send'}</button>
      </form>
    </section>
  )
}
