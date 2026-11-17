import React from 'react';
import { Boxes, Cpu, MessageSquarePlus, Power, Settings, Zap } from 'lucide-react';

export type NavTab = 'chat' | 'models' | 'settings';

interface SidebarProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  online: boolean;
  useGpu: boolean;
  tokensPerSec: number | null;
  onToggleGpu: (next: boolean) => void;
  onToggleDaemon: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  online,
  useGpu,
  tokensPerSec,
  onToggleGpu,
  onToggleDaemon,
}) => {
  const navItems: Array<{ id: NavTab; label: string; icon: React.ReactNode }> = [
    { id: 'chat', label: 'Chat', icon: <MessageSquarePlus className="h-4 w-4" /> },
    { id: 'models', label: 'Models', icon: <Boxes className="h-4 w-4" /> },
    { id: 'settings', label: 'Settings', icon: <Settings className="h-4 w-4" /> },
  ];

  return (
    <aside className="flex w-56 shrink-0 flex-col justify-between border-r border-zinc-800/80 bg-zinc-950 p-3">
      <div>
        <div className="mb-4 flex items-center justify-between px-2 py-1">
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-zinc-100 text-xs font-bold text-zinc-950">
              AI
            </span>
            <span className="text-sm font-semibold tracking-tight text-zinc-100">AI Box</span>
          </div>
          <span
            className={`h-2 w-2 rounded-full ${online ? 'bg-emerald-400' : 'bg-rose-500'}`}
            title={online ? 'Daemon en ligne' : 'Daemon hors ligne'}
          />
        </div>

        <nav className="space-y-1">
          {navItems.map((item) => {
            const active = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onSelectTab(item.id)}
                className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm transition ${
                  active
                    ? 'bg-zinc-800/90 font-medium text-zinc-100'
                    : 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200'
                }`}
              >
                {item.icon}
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>

      <div className="space-y-2 border-t border-zinc-800/80 pt-3">
        <div className="flex items-center justify-between px-1 text-xs text-zinc-400">
          <span>{online ? '127.0.0.1:11434' : 'Hors ligne'}</span>
          {tokensPerSec !== null && (
            <span className="font-medium text-emerald-400">{tokensPerSec.toFixed(1)} tok/s</span>
          )}
        </div>

        <div className="flex gap-1.5">
          <button
            type="button"
            onClick={() => onToggleGpu(!useGpu)}
            disabled={online}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-zinc-800 bg-zinc-900 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
          >
            {useGpu ? <Zap className="h-3.5 w-3.5 text-amber-400" /> : <Cpu className="h-3.5 w-3.5" />}
            {useGpu ? 'GPU' : 'CPU'}
          </button>

          <button
            type="button"
            onClick={onToggleDaemon}
            className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium ${
              online ? 'bg-rose-600/90 text-white hover:bg-rose-500' : 'bg-zinc-100 text-zinc-950 hover:bg-white'
            }`}
          >
            <Power className="h-3.5 w-3.5" />
            {online ? 'Stop' : 'Start'}
          </button>
        </div>
      </div>
    </aside>
  );
};
