import React, { useState } from 'react';
import { ArrowUp, Square } from 'lucide-react';
import { ChatMessage } from '../hooks/useAiboxDaemon';

interface ChatPanelProps {
  messages: ChatMessage[];
  models: string[];
  selectedModel: string;
  isGenerating: boolean;
  online: boolean;
  error: string | null;
  onSelectModel: (model: string) => void;
  onSend: (content: string) => void;
  onStop: () => void;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  models,
  selectedModel,
  isGenerating,
  online,
  error,
  onSelectModel,
  onSend,
  onStop,
}) => {
  const [draft, setDraft] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!draft.trim() || !online || isGenerating) {
      return;
    }
    onSend(draft);
    setDraft('');
  };

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between border-b border-zinc-800/60 px-6 py-3">
        <select
          value={selectedModel}
          onChange={(e) => onSelectModel(e.target.value)}
          disabled={!online || models.length === 0}
          className="rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-1.5 text-xs font-medium text-zinc-200 focus:outline-none disabled:opacity-50"
        >
          {models.length === 0 ? (
            <option value="">Aucun modèle</option>
          ) : (
            models.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))
          )}
        </select>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-3xl space-y-5">
          {messages.length === 0 && (
            <div className="flex h-[55vh] flex-col items-center justify-center text-center">
              <p className="text-lg font-medium text-zinc-200">
                {online ? 'Comment puis-je vous aider ?' : 'Daemon AI Box hors ligne'}
              </p>
              <p className="mt-1 text-xs text-zinc-500">
                {online
                  ? `Modèle actif : ${selectedModel || 'aibox-ternary-demo'}`
                  : 'Cliquez sur Start dans la barre latérale ou lancez aibox serve.'}
              </p>
            </div>
          )}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[82%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-zinc-800 text-zinc-100'
                    : 'text-zinc-200'
                }`}
              >
                {msg.role === 'assistant'
                  ? msg.content.replace('<|channel>thought\n<channel|>', '').trimStart() || '...'
                  : msg.content || '...'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div className="mx-auto mb-2 w-full max-w-3xl px-4">
          <div className="rounded-xl border border-rose-800/70 bg-rose-950/60 px-3.5 py-2 text-xs text-rose-300">
            {error}
          </div>
        </div>
      )}

      <div className="px-4 pb-5 pt-2">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex max-w-3xl items-center gap-2 rounded-2xl border border-zinc-800 bg-zinc-900 px-3 py-2 shadow-lg"
        >
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={online ? 'Envoyer un message...' : 'Daemon hors ligne...'}
            disabled={!online}
            className="flex-1 bg-transparent px-2 py-1.5 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none disabled:opacity-50"
          />
          {isGenerating ? (
            <button
              type="button"
              onClick={onStop}
              className="flex h-8 w-8 items-center justify-center rounded-xl bg-zinc-100 text-zinc-950 hover:bg-white"
              title="Stop"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!online || !draft.trim()}
              className="flex h-8 w-8 items-center justify-center rounded-xl bg-zinc-100 text-zinc-950 hover:bg-white disabled:opacity-30"
              title="Envoyer"
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          )}
        </form>
      </div>
    </div>
  );
};
