import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Chat as ChatMeta, type Paper } from './api'
import { Chat } from './components/Chat'
import { Library } from './components/Library'

export default function App() {
  const [papers, setPapers] = useState<Paper[]>([])
  const [chats, setChats] = useState<ChatMeta[]>([])
  const [activeChat, setActiveChat] = useState<number | null>(null)
  const [pdf, setPdf] = useState<{ paperId: number; page?: number } | null>(null)
  const pollRef = useRef<number | null>(null)

  const refreshPapers = useCallback(async () => {
    const p = await api.papers()
    setPapers(p)
    // poll while anything is still ingesting
    const busy = p.some((x) => x.status === 'pending' || x.status === 'processing')
    if (busy && pollRef.current === null) {
      pollRef.current = window.setTimeout(() => {
        pollRef.current = null
        refreshPapers()
      }, 2000)
    }
  }, [])

  const refreshChats = useCallback(() => api.chats().then(setChats), [])

  useEffect(() => {
    refreshPapers()
    refreshChats()
  }, [refreshPapers, refreshChats])

  const newChat = async () => {
    const c = await api.createChat()
    await refreshChats()
    setActiveChat(c.id)
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>ResearchMind</h1>
        <Library papers={papers} onChanged={refreshPapers}
                 onOpenPdf={(paperId) => setPdf({ paperId })} />
        <section className="chat-list">
          <header>
            <h2>Chats</h2>
            <button onClick={newChat}>+ New</button>
          </header>
          <ul>
            {chats.map((c) => (
              <li key={c.id}>
                <button className={c.id === activeChat ? 'active' : ''}
                        onClick={() => setActiveChat(c.id)}>
                  {c.title}
                </button>
              </li>
            ))}
          </ul>
        </section>
      </aside>

      <main>
        {activeChat !== null ? (
          <Chat chatId={activeChat} onTitleChanged={refreshChats}
                locked={papers.some((p) => p.status === 'pending' || p.status === 'processing')}
                onOpenPdf={(paperId, page) => setPdf({ paperId, page })} />

        ) : (
          <div className="placeholder">
            <p>Upload papers, then start a chat. Answers cite exact pages.</p>
            <button onClick={newChat}>Start a chat</button>
          </div>
        )}
      </main>

      {pdf && (
        <aside className="pdf-panel">
          <button className="pdf-close" onClick={() => setPdf(null)}>× close</button>
          <iframe title="paper" src={api.pdfUrl(pdf.paperId, pdf.page)} key={`${pdf.paperId}-${pdf.page}`} />
        </aside>
      )}
    </div>
  )
}
