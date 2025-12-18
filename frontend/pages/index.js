import { useEffect, useState } from 'react'
import Auth from '../components/Auth'
import Sidebar from '../components/Sidebar'
import Chat from '../components/Chat'

export default function Home(){
  const [token, setToken] = useState(null)
  const [user, setUser] = useState(null)
  const [chats, setChats] = useState([])
  const [currentChatId, setCurrentChatId] = useState(null)
  const [loading, setLoading] = useState(true)

  // Load auth state from localStorage on mount
  useEffect(() => {
    const savedToken = localStorage.getItem('cognitiveai_token')
    const savedUser = localStorage.getItem('cognitiveai_user')

    if (savedToken && savedUser) {
      setToken(savedToken)
      setUser(JSON.parse(savedUser))
    }
    setLoading(false)
  }, [])

  // Load chats when authenticated
  useEffect(() => {
    if (!token) return

    const raw = localStorage.getItem('cognitiveai_chats')
    if (raw) {
      const parsed = JSON.parse(raw)
      setChats(parsed)
      if (parsed.length) setCurrentChatId(parsed[0].id)
    } else {
      const initial = [{ id: 'chat-' + Date.now(), title: 'New chat', messages: [] }]
      setChats(initial)
      setCurrentChatId(initial[0].id)
      localStorage.setItem('cognitiveai_chats', JSON.stringify(initial))
    }
  }, [token])

  const updateChats = (updater) => {
    setChats((prevChats) => {
      const next = typeof updater === 'function' ? updater(prevChats) : updater
      try {
        localStorage.setItem('cognitiveai_chats', JSON.stringify(next))
      } catch (e) {
        console.warn('Failed to persist chats', e)
      }
      return next
    })
  }

  const handleAuthSuccess = (newToken, userInfo) => {
    setToken(newToken)
    setUser(userInfo)
  }

  const handleLogout = () => {
    setToken(null)
    setUser(null)
    setChats([])
    setCurrentChatId(null)
    localStorage.removeItem('cognitiveai_token')
    localStorage.removeItem('cognitiveai_user')
    localStorage.removeItem('cognitiveai_chats')
  }

  if (loading) {
    return <div className="loading">Loading...</div>
  }

  if (!token) {
    return <Auth onAuthSuccess={handleAuthSuccess} />
  }

  return (
  <div className="app-root">
    <Sidebar
      chats={chats}
      currentChatId={currentChatId}
      setCurrentChatId={setCurrentChatId}
      updateChats={updateChats}
      user={user}
      onLogout={handleLogout}
      token={token}
    />
    <Chat
      chats={chats}
      currentChatId={currentChatId}
      updateChats={updateChats}
      token={token}
    />
  </div>
)
}
