import React from 'react';
import { Box, CheckCircle2 } from 'lucide-react';

interface ModelsPanelProps {
  models: string[];
  selectedModel: string;
  online: boolean;
  onSelectModel: (model: string) => void;
}

export const ModelsPanel: React.FC<ModelsPanelProps> = ({
  models,
  selectedModel,
  online,
  onSelectModel,
}) => {
  return (
    <section className="mx-auto w-full max-w-3xl p-8">
      <h1 className="mb-6 text-xl font-semibold text-zinc-100">Models</h1>

      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/70 p-4">
        {!online ? (
          <p className="py-6 text-center text-sm text-zinc-400">
            Démarrez le daemon AI Box pour lister les modèles locaux.
          </p>
        ) : models.length === 0 ? (
          <p className="py-6 text-center text-sm text-zinc-400">
            Aucun modèle détecté dans <code className="text-zinc-200">~/.aibox/models</code>.
          </p>
        ) : (
          <ul className="divide-y divide-zinc-800/80">
            {models.map((name) => {
              const isSelected = name === selectedModel;
              return (
                <li key={name} className="flex items-center justify-between py-3">
                  <div className="flex items-center gap-3">
                    <Box className="h-4 w-4 text-zinc-400" />
                    <span className="text-sm font-medium text-zinc-100">{name}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => onSelectModel(name)}
                    className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      isSelected
                        ? 'bg-zinc-100 text-zinc-950'
                        : 'border border-zinc-700 bg-zinc-950 text-zinc-300 hover:bg-zinc-800'
                    }`}
                  >
                    {isSelected && <CheckCircle2 className="h-3.5 w-3.5" />}
                    {isSelected ? 'Actif' : 'Utiliser'}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
};
