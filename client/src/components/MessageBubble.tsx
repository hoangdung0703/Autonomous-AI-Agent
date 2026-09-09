import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Source, Step } from '../types';
import ReasoningStep from './ReasoningStep';

export interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  content: string;
  steps?: Step[];
  sources?: Source[];
}

interface MessageBubbleProps {
  message: ChatMessage;
}

function MessageBubble({ message }: MessageBubbleProps) {
  if (message.role === 'user') {
    return (
      <div className="message-row message-row-user">
        <div className="message-bubble message-bubble-user">{message.content}</div>
      </div>
    );
  }

  return (
    <div className="message-row message-row-agent">
      <div className="message-bubble message-bubble-agent">
        {message.steps && message.steps.length > 0 && (
          <div className="reasoning-steps">
            {message.steps.map((step, index) => (
              <ReasoningStep key={index} step={step} />
            ))}
          </div>
        )}

        <div className="message-answer">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="source-chips">
            {message.sources.map((source, index) => (
              <span key={index} className="source-chip">
                {source.document_name} · {source.section_title}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default MessageBubble;
