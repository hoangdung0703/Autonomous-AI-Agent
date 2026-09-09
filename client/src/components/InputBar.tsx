import { useState } from 'react';
import type { KeyboardEvent } from 'react';

interface InputBarProps {
  onSend: (question: string) => void;
  disabled: boolean;
}

function InputBar({ onSend, disabled }: InputBarProps) {
  const [value, setValue] = useState('');

  const submit = () => {
    const question = value.trim();
    if (!question || disabled) return;
    onSend(question);
    setValue('');
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="input-bar">
      <input
        type="text"
        className="input-bar-field"
        placeholder="Ask about HR, legal, or ops policies..."
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      <button
        type="button"
        className="input-bar-send"
        onClick={submit}
        disabled={disabled || !value.trim()}
      >
        Send
      </button>
    </div>
  );
}

export default InputBar;
