import React from 'react';
import { Folder, SlidersHorizontal, Zap } from 'lucide-react';

interface SettingsPanelProps {
  useGpu: boolean;
  online: boolean;
  maxTokens: number;
  onToggleGpu: (next: boolean) => void;
  onChangeMaxTokens: (next: number) => void;
  onResetDefaults: () => void;
}

export const SettingsPanel: React.FC<SettingsPanelProps> = ({
  useGpu,
  online,
  maxTokens,
  onToggleGpu,
  onChangeMaxTokens,
  onResetDefaults,
}) => {
  const steps = [256, 512, 1024, 2048, 4096];

  return (
    <section className="mx-auto w-full max-w-3xl p-8">
      <h1 className="mb-6 text-xl font-semibold text-zinc-100">Settings</h1>

      <div className="divide-y divide-zinc-800/80 rounded-2xl border border-zinc-800 bg-zinc-900/70 px-6">
        <div className="flex items-center justify-between py-5">
          <div className="flex items-start gap-3">
            <Zap className="mt-0.5 h-4 w-4 text-zinc-400" />
            <div>
              <p className="text-sm font-medium text-zinc-100">GPU Acceleration (CUDA)</p>
              <p className="text-xs text-zinc-400">
                Exécuter les kernels Q1/Q2/Q4 sur GPU plutôt que sur CPU AVX2.
              </p>
            </div>
          </div>
          <button
            type="button"
            disabled={online}
            onClick={() => onToggleGpu(!useGpu)}
            className={`relative h-6 w-11 rounded-full transition disabled:opacity-50 ${
              useGpu ? 'bg-zinc-100' : 'bg-zinc-700'
            }`}
          >
            <span
              className={`block h-5 w-5 rounded-full transition-transform ${
                useGpu ? 'translate-x-5 bg-zinc-950' : 'translate-x-0.5 bg-zinc-200'
              }`}
            />
          </button>
        </div>

        <div className="py-5">
          <div className="mb-3 flex items-start gap-3">
            <Folder className="mt-0.5 h-4 w-4 text-zinc-400" />
            <div>
              <p className="text-sm font-medium text-zinc-100">Model location</p>
              <p className="text-xs text-zinc-400">Répertoire local des modèles SafeTensors.</p>
            </div>
          </div>
          <div className="rounded-xl border border-zinc-800 bg-zinc-950 px-3.5 py-2 text-xs text-zinc-300">
            ~/.aibox/models
          </div>
        </div>

        <div className="py-5">
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-start gap-3">
              <SlidersHorizontal className="mt-0.5 h-4 w-4 text-zinc-400" />
              <div>
                <p className="text-sm font-medium text-zinc-100">Max generation tokens</p>
                <p className="text-xs text-zinc-400">Nombre maximal de tokens générés par réponse.</p>
              </div>
            </div>
            <span className="rounded-md bg-zinc-800 px-2 py-0.5 text-xs font-medium text-zinc-200">
              {maxTokens}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={steps.length - 1}
            value={Math.max(0, steps.indexOf(maxTokens))}
            onChange={(e) => onChangeMaxTokens(steps[Number(e.target.value)] ?? 512)}
            className="w-full accent-zinc-100"
          />
          <div className="mt-1 flex justify-between text-[11px] text-zinc-500">
            {steps.map((s) => (
              <span key={s}>{s}</span>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-4 flex justify-end">
        <button
          type="button"
          onClick={onResetDefaults}
          className="rounded-lg bg-zinc-100 px-4 py-2 text-xs font-medium text-zinc-950 hover:bg-white"
        >
          Reset to defaults
        </button>
      </div>
    </section>
  );
};
