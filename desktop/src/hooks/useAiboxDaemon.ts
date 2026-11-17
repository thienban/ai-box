import { useCallback, useEffect, useRef, useState } from 'react';

const DAEMON_CANDIDATES = ['http://127.0.0.1:11435', 'http://127.0.0.1:11434'];

const findAiboxDaemonUrl = async (): Promise<string | null> => {
  for (const baseUrl of DAEMON_CANDIDATES) {
    try {
      const res = await fetch(`${baseUrl}/health`);
      if (!res.ok) {
        continue;
      }
      const text = await res.text();
      if (text.includes('aibox is running')) {
        return baseUrl;
      }
    } catch {
      continue;
    }
  }
  return null;
};

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

interface DaemonStatus {
  online: boolean;
  models: string[];
  selectedModel: string;
  tokensPerSec: number | null;
}

const isTauriEnv = (): boolean =>
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

const invokeTauri = async <T,>(cmd: string): Promise<T | null> => {
  if (!isTauriEnv()) {
    return null;
  }
  const { invoke } = await import('@tauri-apps/api/core');
  return invoke<T>(cmd);
};

const parseSseDelta = (line: string): string => {
  const cleaned = line.trim();
  if (!cleaned.startsWith('data:') || cleaned === 'data: [DONE]') {
    return '';
  }
  const jsonStr = cleaned.slice(5).trim();
  if (!jsonStr) {
    return '';
  }
  const parsed = JSON.parse(jsonStr) as {
    choices?: Array<{ delta?: { content?: string } }>;
  };
  return parsed.choices?.[0]?.delta?.content ?? '';
};

const readSseStream = async (
  body: ReadableStream<Uint8Array>,
  onDelta: (delta: string) => void
): Promise<{ tokenCount: number; firstTokenAt: number | null }> => {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let tokenCount = 0;
  let firstTokenAt: number | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      return { tokenCount, firstTokenAt };
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      const delta = parseSseDelta(line);
      if (delta) {
        if (firstTokenAt === null) {
          firstTokenAt = performance.now();
        }
        tokenCount += 1;
        onDelta(delta);
      }
    }
  }
};

export const useAiboxDaemon = () => {
  const [status, setStatus] = useState<DaemonStatus>({
    online: false,
    models: [],
    selectedModel: '',
    tokensPerSec: null,
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [useGpu, setUseGpu] = useState(true);
  const [maxTokens, setMaxTokens] = useState(512);
  const [daemonUrl, setDaemonUrl] = useState(DAEMON_CANDIDATES[1]);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refreshStatus = useCallback(async () => {
    const activeUrl = await findAiboxDaemonUrl();
    if (!activeUrl) {
      setStatus((prev) => ({ ...prev, online: false, models: [] }));
      return;
    }
    setDaemonUrl(activeUrl);
    try {
      const res = await fetch(`${activeUrl}/api/tags`);
      if (!res.ok) {
        setStatus((prev) => ({ ...prev, online: false }));
        return;
      }
      const data = (await res.json()) as { models?: Array<{ name: string }> };
      const models = (data.models ?? []).map((m) => m.name);
      setStatus((prev) => ({
        ...prev,
        online: true,
        models,
        selectedModel: models.includes(prev.selectedModel) ? prev.selectedModel : (models[0] ?? ''),
      }));
      setError(null);
    } catch {
      setStatus((prev) => ({ ...prev, online: false }));
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    const timer = setInterval(() => void refreshStatus(), 4000);
    return () => clearInterval(timer);
  }, [refreshStatus]);

  const toggleDaemon = useCallback(async () => {
    try {
      setError(null);
      if (status.online) {
        await invokeTauri('stop_daemon');
      } else {
        await invokeTauri(useGpu ? 'start_daemon_gpu' : 'start_daemon_cpu');
      }
      setTimeout(() => void refreshStatus(), 600);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [status.online, useGpu, refreshStatus]);

  const selectModel = useCallback((model: string) => {
    setStatus((prev) => ({ ...prev, selectedModel: model }));
  }, []);

  const stopGeneration = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsGenerating(false);
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      const trimmed = content.trim();
      if (!trimmed || isGenerating) {
        return;
      }
      setError(null);
      const userMsg: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: trimmed };
      const assistantId = crypto.randomUUID();
      const nextHistory = [...messages, userMsg];
      setMessages([...nextHistory, { id: assistantId, role: 'assistant', content: '' }]);
      setIsGenerating(true);

      const controller = new AbortController();
      abortRef.current = controller;
      const startedAt = performance.now();

      try {
        const res = await fetch(`${daemonUrl}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            model: status.selectedModel || 'aibox-ternary-demo',
            messages: nextHistory.map((m) => ({ role: m.role, content: m.content })),
            max_tokens: maxTokens,
            stream: true,
          }),
        });
        if (!res.ok || !res.body) {
          throw new Error(`Erreur HTTP ${res.status}`);
        }
        const { tokenCount, firstTokenAt } = await readSseStream(res.body, (delta) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + delta } : m))
          );
        });
        const decodeStart = firstTokenAt ?? startedAt;
        const decodeTokens = tokenCount > 1 && firstTokenAt !== null ? tokenCount - 1 : tokenCount;
        const elapsedSec = Math.max((performance.now() - decodeStart) / 1000, 0.001);
        setStatus((prev) => ({ ...prev, tokensPerSec: decodeTokens / elapsedSec }));
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        setIsGenerating(false);
        abortRef.current = null;
      }
    },
    [daemonUrl, isGenerating, maxTokens, messages, status.selectedModel]
  );

  return {
    status,
    messages,
    isGenerating,
    useGpu,
    setUseGpu,
    maxTokens,
    setMaxTokens,
    error,
    toggleDaemon,
    selectModel,
    sendMessage,
    stopGeneration,
  };
};
