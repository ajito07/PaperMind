const API = 'http://localhost:8000'

export interface Paper {
  id: number
  title: string
  authors: string | null
  year: number | null
  num_pages: number | null
  status: 'pending' | 'processing' | 'ready' | 'failed'
  error: string | null
}

export interface Citation {
  marker: number
  chunk_id: number
  paper_id: number
  title: string
  page_start: number
  page_end: number
  snippet: string
}

export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
}

export interface Chat {
  id: number
  title: string
}

export interface TraceStage {
  stage: string
  duration_ms: number
  payload: Record<string, unknown>
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(API + path, init)
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? r.statusText)
  return r.json()
}

export const api = {
  papers: () => json<Paper[]>('/papers'),
  deletePaper: (id: number) => fetch(`${API}/papers/${id}`, { method: 'DELETE' }),
  pdfUrl: (id: number, page?: number) => `${API}/papers/${id}/pdf${page ? `#page=${page}` : ''}`,
  upload: (file: File) => {
    const body = new FormData()
    body.append('file', file)
    return json<{ id: number }>('/papers', { method: 'POST', body })
  },
  chats: () => json<Chat[]>('/chats'),
  createChat: () => json<Chat>('/chats', { method: 'POST' }),
  chat: (id: number) => json<Chat & { messages: Message[] }>(`/chats/${id}`),
  trace: (messageId: number) => json<TraceStage[]>(`/messages/${messageId}/trace`),
}

export type StreamEvent =
  | { type: 'token'; content: string }
  | { type: 'citations'; citations: Citation[] }
  | { type: 'done'; message_id: number }
  | { type: 'error'; message: string }

export async function streamMessage(
  chatId: number,
  content: string,
  onEvent: (ev: StreamEvent) => void,
): Promise<void> {
  const r = await fetch(`${API}/chats/${chatId}/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ content }),
  })
  if (!r.ok) throw new Error((await r.json().catch(() => null))?.detail ?? r.statusText)
  if (!r.body) throw new Error('no response body')
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)))
    }
  }
}
