import { useState } from 'react';
import './App.css';
import ChatWindow from './components/ChatWindow';
import InputBar from './components/InputBar';
import type { ChatMessage } from './components/MessageBubble';
import { postChat } from './services/api';

function App() {
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async (question: string) => {
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: question,
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await postChat(question, conversationId);
      setConversationId(response.conversation_id);
      setMessages((prev) => [
        ...prev,
        {
          id: `agent-${Date.now()}`,
          role: 'agent',
          content: response.answer,
          steps: response.steps,
          sources: response.sources,
        },
      ]);
    } catch (error) {
      console.error('Chat request failed:', error);
      const message =
        error instanceof Error
          ? error.message
          : 'Something went wrong while the agent was thinking. Please try again.';
      setMessages((prev) => [
        ...prev,
        {
          id: `agent-error-${Date.now()}`,
          role: 'agent',
          content: message,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>VinTech Enterprise Knowledge Agent</h1>
      </header>
      <ChatWindow messages={messages} isLoading={isLoading} />
      <InputBar onSend={handleSend} disabled={isLoading} />
    </div>
  );
}

export default App;
