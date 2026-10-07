import React, { useEffect, useState } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { checkHealth } from '../services/api';
import { HealthResponse } from '../types';

export const HealthBadge: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<boolean>(false);

  const fetchStatus = async () => {
    setLoading(true);
    setError(false);
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !health) {
    return (
      <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-slate-800/80 text-slate-400 border border-slate-700">
        <RefreshCw className="w-3 h-3 animate-spin text-slate-400" />
        <span>Connecting...</span>
      </div>
    );
  }

  if (error || !health || health.status !== 'healthy') {
    return (
      <button
        onClick={fetchStatus}
        title="Click to retry connection"
        className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-rose-500/10 text-rose-400 border border-rose-500/30 hover:bg-rose-500/20 transition-all cursor-pointer"
      >
        <AlertCircle className="w-3 h-3 text-rose-400" />
        <span>Backend Disconnected</span>
      </button>
    );
  }

  return (
    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 shadow-sm shadow-emerald-950/40">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
      </span>
      <span className="font-semibold tracking-wide">API v{health.version} Online</span>
    </div>
  );
};
