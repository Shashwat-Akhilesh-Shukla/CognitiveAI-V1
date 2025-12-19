import React, { useEffect, useRef, useState } from 'react'
import Message from './Message'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL
console.log('BACKEND_URL in Chat.js:', BACKEND_URL)

export default function Chat({ chats, currentChatId, updateChats, token }){
  const [text, setText] = useState('')
  const [attachingFile, setAttachingFile] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState('')
  const [fileInfo, setFileInfo] = useState(null)
  const inputRef = useRef(null)
  const messagesRef = useRef(null)

  const current = chats.find(c=> c.id === currentChatId) || { id: null, messages: [] }

  useEffect(()=>{ if(currentChatId && !current) console.warn('No current chat') }, [currentChatId])

  function pushMessage(role, content, extra={}){
    updateChats(prev => prev.map(c=> c.id===currentChatId? {...c, messages: [...c.messages, { id: 'm-'+Date.now(), role, content, ...extra }]}: c))
  }

  useEffect(()=>{
    // auto-scroll to bottom when messages change
    const el = messagesRef.current
    if(el) el.scrollTop = el.scrollHeight
  }, [chats, currentChatId])

  async function send(){
    console.log('[send] called', { text, fileInfo: fileInfo ? { filename: fileInfo.filename } : null })
    if(!text && !fileInfo) return

    // Build message to display in UI (do NOT include extracted text)
    const displayedMessage = text || (fileInfo ? `Uploaded file: ${fileInfo.filename}` : '')
    pushMessage('user', displayedMessage, { file: fileInfo })
    setText('')

    // Prepare payload for backend — include doc_id so server can attach extracted text (kept server-side)
    const payload = { message: displayedMessage }
    if(fileInfo && fileInfo.doc_id){ payload.doc_id = fileInfo.doc_id }

    // Call backend chat with authorization token
    try{
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 
          'Content-Type':'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      })
      const j = await res.json()
      if(!res.ok){
        pushMessage('ai', 'Error: ' + (j.detail || j.message || 'Chat failed'))
        return
      }
      pushMessage('ai', j.response || JSON.stringify(j))
    }catch(err){
      pushMessage('ai', 'Error: '+String(err))
    }
  }

  function uploadFileWithProgress(file, onProgress){
    return new Promise((resolve, reject)=>{
      console.log('[uploadFileWithProgress] starting', { filename: file.name, url: `${BACKEND_URL}/upload_pdf` })
      const form = new FormData()
      form.append('file', file)

      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BACKEND_URL}/upload_pdf`)
      xhr.setRequestHeader('Authorization', `Bearer ${token}`)
      console.log('[uploadFileWithProgress] XHR opened')

      xhr.upload.onprogress = function(e){
        if(e.lengthComputable){
          const pct = Math.round((e.loaded / e.total) * 100)
          console.log('[xhr.progress]', pct + '%')
          onProgress(pct)
        } else {
          console.log('[xhr.progress] indeterminate')
          onProgress(5)
        }
      }

      xhr.onload = function(){
        console.log('[xhr.onload]', xhr.status, xhr.responseText.substring(0, 100))
        if(xhr.status >= 200 && xhr.status < 300){
          try{ onProgress(100) } catch(e){}
          try{ const json = JSON.parse(xhr.responseText); resolve(json) } catch(e){ resolve({ status: 'processing' }) }
        } else {
          reject(new Error(`Upload failed (${xhr.status})`))
        }
      }

      xhr.onerror = function(){ 
        console.error('[xhr.onerror]')
        reject(new Error('Network error during upload')) 
      }
      console.log('[xhr.send] sending form...')
      xhr.send(form)
    })
  }

  function onFileChange(e){
    const f = e.target.files?.[0]
    if(f){
      console.log('[onFileChange] file selected:', f.name)
      setAttachingFile(f)
      // Immediately start uploading in the background
        uploadFileWithProgress(f, (pct)=> setUploadProgress(pct))
          .then(j => {
            console.log('[onFileChange] upload completed')
            // store doc_id returned by backend; do NOT store extracted text in frontend
            setFileInfo({ 
              filename: f.name, 
              uploadStatus: j.status || 'processing',
              doc_id: j.doc_id || null
            })
          })
        .catch(err => {
          const msg = err && err.message? err.message : String(err)
          console.error('[onFileChange] upload failed:', msg)
          setError('Upload failed: ' + msg)
        })
        .finally(() => {
          setUploadProgress(0)
        })
    }
  }

  return (
    <main className="chat-main">
      <div className="messages" ref={messagesRef}>
        {current.messages && current.messages.map(m=> <Message key={m.id} m={m} />)}
      </div>

      {/* Attachment bar sits just above the composer, similar to ChatGPT */}
      {attachingFile && (
        <div className="attachment-bar">
          <div
            className="progress-circle"
            style={{ background: `conic-gradient(var(--accent) ${uploadProgress * 3.6}deg, rgba(255,255,255,0.06) ${uploadProgress * 3.6}deg)` }}
          >
            <div className="progress-inner">{uploadProgress}%</div>
          </div>
          <div className="attachment-name">{attachingFile.name}</div>
          <div style={{flex:1}} />
          <button className="btn small" onClick={()=>{ setAttachingFile(null); setUploadProgress(0); setFileInfo(null) }}>Remove</button>
        </div>
      )}

      <div className="composer">
        <label className="attach">
          📎
          <input type="file" accept="application/pdf" onChange={onFileChange} />
        </label>
        {error && (
          <div className="toast error">
            {error} <button onClick={()=>setError('')} className="btn small">Dismiss</button>
          </div>
        )}
        <input
          ref={inputRef}
          className="text-input"
          value={text}
          onChange={(e)=>setText(e.target.value)}
          placeholder={attachingFile? `Attached: ${attachingFile.name}` : 'Type a message...'}
          onKeyDown={(e)=>{ if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); send() } }}
        />
        <button className="btn send" onClick={send}>Send</button>
      </div>
    </main>
  )
}
