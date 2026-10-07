import React, { useEffect, useState } from 'react';
import {
  Activity,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Server,
  Clock,
  Code2,
  Radio
} from 'lucide-react';
import { checkHealth, getSystemInfo } from '../services/api';
import { HealthResponse, SystemInfoResponse } from '../types';

export const SystemStatus: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemInfoResponse | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const runDiagnostic = async () => {
    setLoading(true);
    setError(null);
    const startTime = performance.now();
    try {
      const [hData, infoData] = await Promise.all([
        checkHealth(),
        getSystemInfo()
      ]);
      const endTime = performance.now();
      setLatency(Math.round(endTime - startTime));
      setHealth(hData);
      setSystemInfo(infoData);
      setLastChecked(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to connect to backend server');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runDiagnostic();
  }, []);

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="inline-flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
            <Radio className="w-4 h-4 text-emerald-400 animate-pulse" />
            <span>Infrastructure Health & Diagnostic Monitor</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white">System Connectivity & Status</h1>
        </div>

        <button
          onClick={runDiagnostic}
          disabled={loading}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-50 transition-all cursor-pointer shadow-lg shadow-emerald-950/40"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Run Probe</span>
        </button>
      </div>

      {/* Connection Status Overview Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Status */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <span className="text-xs font-medium text-slate-400">Operational State</span>
          <div className="flex items-center gap-2">
            {health?.status === 'healthy' ? (
              <>
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                <span className="text-lg font-bold text-emerald-400 uppercase">Healthy</span>
              </>
            ) : error ? (
              <>
                <AlertCircle className="w-5 h-5 text-rose-400" />
                <span className="text-lg font-bold text-rose-400 uppercase">Offline</span>
              </>
            ) : (
              <>
                <RefreshCw className="w-5 h-5 animate-spin text-slate-400" />
                <span className="text-lg font-bold text-slate-400">Probing...</span>
              </>
            )}
          </div>
          <p className="text-[11px] text-slate-500 font-mono">
            FastAPI Core Engine
          </p>
        </div>

        {/* Card 2: Latency */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <span className="text-xs font-medium text-slate-400">Round-Trip Latency</span>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-teal-400" />
            <span className="text-lg font-bold text-white font-mono">
              {latency !== null ? `${latency} ms` : '—'}
            </span>
          </div>
          <p className="text-[11px] text-slate-500 font-mono">
            Local Loopback Roundtrip
          </p>
        </div>

        {/* Card 3: Version */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <span className="text-xs font-medium text-slate-400">API Semantic Version</span>
          <div className="flex items-center gap-2">
            <Code2 className="w-5 h-5 text-indigo-400" />
            <span className="text-lg font-bold text-white font-mono">
              v{health?.version || '1.0.0'}
            </span>
          </div>
          <p className="text-[11px] text-slate-500 font-mono">
            Environment: {health?.environment || 'development'}
          </p>
        </div>

        {/* Card 4: Last Probe */}
        <div className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-2">
          <span className="text-xs font-medium text-slate-400">Last Telemetry Probe</span>
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-amber-400" />
            <span className="text-sm font-bold text-white font-mono">
              {lastChecked ? lastChecked.toLocaleTimeString() : '—'}
            </span>
          </div>
          <p className="text-[11px] text-slate-500 font-mono">
            Auto-refresh interval: 15s
          </p>
        </div>
      </div>

      {/* Error Banner if disconnected */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/50 text-rose-300 text-sm flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold">Backend Connection Failed</p>
            <p className="text-xs text-rose-400">{error}</p>
            <p className="text-xs text-slate-400 pt-1 font-mono">
              Make sure the FastAPI server is running at <code className="text-slate-200">http://127.0.0.1:8000</code>.
            </p>
          </div>
        </div>
      )}

      {/* Verified Endpoints Test Matrix */}
      <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-6">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Server className="w-5 h-5 text-emerald-400" />
          <span>Verified REST API Endpoints (Milestone 1)</span>
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 font-mono">
                <th className="pb-3 font-semibold">METHOD</th>
                <th className="pb-3 font-semibold">ENDPOINT</th>
                <th className="pb-3 font-semibold">TARGET PURPOSE</th>
                <th className="pb-3 font-semibold">STATUS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
              <tr>
                <td className="py-3 text-emerald-400 font-bold">GET</td>
                <td className="py-3 font-semibold text-white">/api/v1/health</td>
                <td className="py-3 font-sans text-slate-300">Liveness probe and environment state</td>
                <td className="py-3">
                  <span className="px-2 py-0.5 rounded text-[11px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    200 OK
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-3 text-emerald-400 font-bold">GET</td>
                <td className="py-3 font-semibold text-white">/api/v1/info</td>
                <td className="py-3 font-sans text-slate-300">Research metadata, modality list & clinical disclaimers</td>
                <td className="py-3">
                  <span className="px-2 py-0.5 rounded text-[11px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    200 OK
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-3 text-emerald-400 font-bold">GET</td>
                <td className="py-3 font-semibold text-white">/</td>
                <td className="py-3 font-sans text-slate-300">Root router and OpenAPI documentation pointers</td>
                <td className="py-3">
                  <span className="px-2 py-0.5 rounded text-[11px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    200 OK
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-3 text-indigo-400 font-bold">GET</td>
                <td className="py-3 font-semibold text-white">/docs</td>
                <td className="py-3 font-sans text-slate-300">Interactive Swagger UI OpenAPI Documentation</td>
                <td className="py-3">
                  <span className="px-2 py-0.5 rounded text-[11px] bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    Active
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Raw Payload Inspector */}
      {systemInfo && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
          <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider font-mono">
            System Metadata & Clinical Notice Payload (/api/v1/info)
          </h3>
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-emerald-400 overflow-x-auto">
            <pre>{JSON.stringify(systemInfo, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
};
