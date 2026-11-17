import React, { useState } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { ModelsPanel } from './components/ModelsPanel';
import { SettingsPanel } from './components/SettingsPanel';
import { NavTab, Sidebar } from './components/Sidebar';
import { useAiboxDaemon } from './hooks/useAiboxDaemon';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>('chat');
  const {
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
  } = useAiboxDaemon();

  return (
    <main className="flex h-screen bg-[#141414] text-zinc-100">
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        online={status.online}
        useGpu={useGpu}
        tokensPerSec={status.tokensPerSec}
        onToggleGpu={setUseGpu}
        onToggleDaemon={toggleDaemon}
      />

      <div className="flex flex-1 flex-col overflow-hidden">
        {activeTab === 'chat' && (
          <ChatPanel
            messages={messages}
            models={status.models}
            selectedModel={status.selectedModel}
            isGenerating={isGenerating}
            online={status.online}
            error={error}
            onSelectModel={selectModel}
            onSend={sendMessage}
            onStop={stopGeneration}
          />
        )}
        {activeTab === 'models' && (
          <ModelsPanel
            models={status.models}
            selectedModel={status.selectedModel}
            online={status.online}
            onSelectModel={(model) => {
              selectModel(model);
              setActiveTab('chat');
            }}
          />
        )}
        {activeTab === 'settings' && (
          <SettingsPanel
            useGpu={useGpu}
            online={status.online}
            maxTokens={maxTokens}
            onToggleGpu={setUseGpu}
            onChangeMaxTokens={setMaxTokens}
            onResetDefaults={() => {
              setUseGpu(true);
              setMaxTokens(512);
            }}
          />
        )}
      </div>
    </main>
  );
};
