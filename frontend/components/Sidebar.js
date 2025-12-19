import { nanoid } from 'nanoid'
import React, { useEffect, useState } from 'react'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL

export default function Sidebar({ chats, currentChatId, setCurrentChatId, updateChats, user, onLogout, token }){
  function createNew(){
    // store current as-is and create new
    const newChat = { id: 'chat-'+nanoid(6), title: 'New chat', messages: [] }
    updateChats((prev)=>[newChat, ...prev])
    setCurrentChatId(newChat.id)
  }

  function selectChat(id){
    setCurrentChatId(id)
  }

  function renameChat(id, title){
    updateChats(prev=>prev.map(c=> c.id===id? {...c, title}: c))
  }

  function deleteChat(id){
    updateChats(prev=>{
      const next = prev.filter(c=> c.id !== id)
      // if no chats remain, create a new one
      if(next.length === 0){
        const fresh = { id: 'chat-'+nanoid(6), title: 'New chat', messages: [] }
        setCurrentChatId(fresh.id)
        return [fresh]
      }

      // if deleting the current chat, move selection to first
      if(id === currentChatId){
        setCurrentChatId(next[0].id)
      }

      return next
    })
  }

  async function handleLogout() {
    try {
      // Call logout endpoint to clear server-side STM
      await fetch(`${BACKEND_URL}/auth/logout`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      })
    } catch (e) {
      console.error('Logout request failed:', e)
    }
    onLogout()
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h3>CognitiveAI</h3>
        <div className="sidebar-actions">
          <button onClick={createNew} className="btn">+ New Chat</button>
          <DarkToggle />
        </div>
      </div>

      <div className="chat-list">
        {chats.map(chat=> (
          <div key={chat.id} className={`chat-item ${chat.id===currentChatId? 'active':''}`} onClick={()=>selectChat(chat.id)}>
            <input
              className="chat-title"
              value={chat.title}
              onChange={(e)=>renameChat(chat.id, e.target.value)}
            />
            <div className="chat-meta">{chat.messages.length} msgs</div>
            <button className="btn delete" onClick={(e)=>{ e.stopPropagation(); deleteChat(chat.id) }}>Delete</button>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        {user && (
          <div className="user-info">
            <p className="user-label">Logged in as:</p>
            <p className="username">{user.username}</p>
          </div>
        )}
        <button onClick={handleLogout} className="btn logout">Logout</button>
      </div>
    </aside>
  )
}

function DarkToggle(){
  const [dark, setDark] = useState(false)
  const [mounted, setMounted] = useState(false)

  useEffect(()=>{
    // run only on client to avoid SSR/content mismatch
    try{
      const saved = localStorage.getItem('cognitiveai_dark')
      const initial = saved === 'true'
      setDark(initial)
      document.documentElement.classList.toggle('dark', initial)
      document.documentElement.classList.toggle('light', !initial)
    }catch(e){
      // ignore
    }
    setMounted(true)
  }, [])

  function toggle(){
    const next = !dark
    setDark(next)
    try{
      localStorage.setItem('cognitiveai_dark', String(next))
      document.documentElement.classList.toggle('dark', next)
      document.documentElement.classList.toggle('light', !next)
    }catch(e){ }
  }

  // during SSR render nothing to avoid mismatch
  const label = !mounted ? '...' : (dark ? 'Light' : 'Dark')
  return <button className="btn toggle" onClick={toggle}>{label}</button>
}
