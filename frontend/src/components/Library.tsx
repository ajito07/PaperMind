import { useRef, useState } from 'react'
import { api, type Paper } from '../api'

export function Library({ papers, onChanged, onOpenPdf }: {
  papers: Paper[]
  onChanged: () => void
  onOpenPdf: (paperId: number) => void
}) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [uploadError, setUploadError] = useState('')

  const upload = async (file: File) => {
    setUploadError('')
    try {
      await api.upload(file)
      onChanged()
    } catch (e) {
      setUploadError(String((e as Error).message || e))
    }
  }

  return (
    <section className="library">
      <header>
        <h2>Library</h2>
        <button onClick={() => fileRef.current?.click()}>+ PDF</button>
        <input
          ref={fileRef} type="file" accept=".pdf" hidden
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) upload(f)
            e.target.value = ''
          }}
        />
      </header>
      {uploadError && <p className="error">{uploadError}</p>}
      <ul>
        {papers.map((p) => (
          <li key={p.id} className={`paper ${p.status}`}>
            <button className="paper-title" disabled={p.status !== 'ready'}
                    onClick={() => onOpenPdf(p.id)} title={p.title}>
              {p.title}
            </button>
            <span className="paper-meta">
              {p.status === 'ready'
                ? `${p.authors?.split(',')[0] ?? '?'} · ${p.num_pages} pp`
                : p.status === 'failed' ? `failed: ${p.error}` : p.status + '…'}
            </span>
            <button className="paper-delete" title="Delete"
                    onClick={() => api.deletePaper(p.id).then(onChanged)}>×</button>
          </li>
        ))}
        {papers.length === 0 && <li className="empty">Upload a paper to get started.</li>}
      </ul>
    </section>
  )
}
