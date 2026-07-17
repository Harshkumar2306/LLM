import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles } from 'lucide-react';
import './index.css';

// This URL will need to be replaced with the Hugging Face Space URL once deployed!
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/generate';

function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', content: "Welcome! I am your custom LLM trained on Shakespeare.\n\nType a prompt below (e.g. 'ROMEO:') to see what I generate!" }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userPrompt = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userPrompt }]);
    setIsLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userPrompt, max_new_tokens: 200 })
      });

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      setMessages(prev => [...prev, { role: 'bot', content: data.response }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'bot', 
        content: `Oops! There was an error connecting to the model.\n\nError: ${error.message}\n\n(Did you remember to start the backend server?)` 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1><Sparkles size={32} style={{ display: 'inline', marginRight: '10px', verticalAlign: 'bottom' }}/> SwiGLU LLM</h1>
        <div className="subtitle">17M Parameters • Custom Architecture</div>
      </header>

      <div className="chat-container">
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-label">{msg.role === 'user' ? 'You' : 'Model'}</div>
              <div className="message-content">
                {msg.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="message-label">Model</div>
              <div className="message-content">
                <div className="loading-dots">
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSend} className="input-area">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a Shakespearean prompt..."
            disabled={isLoading}
            autoComplete="off"
          />
          <button type="submit" disabled={!input.trim() || isLoading}>
            <Send size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
