import React, { useState, useEffect } from 'react';
import {
  BarChart2,
  RefreshCw,
  Shield,
  ChevronDown,
  ChevronUp,
  Calculator,
  User,
  AlertCircle
} from 'lucide-react';
import { getRoomAnalysis } from '../services/api';
import { CommunicationMetrics } from '../types';

interface Props {
  initialRoomId?: string;
}

export const CommunicationAnalysisView: React.FC<Props> = ({ initialRoomId = 'TEST001' }) => {
  const [targetRoom, setTargetRoom] = useState<string>(initialRoomId);
  const [metrics, setMetrics] = useState<CommunicationMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [showCalculations, setShowCalculations] = useState<boolean>(true);

  const fetchAnalysis = async (roomToFetch: string) => {
    if (!roomToFetch.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getRoomAnalysis(roomToFetch.trim().toUpperCase());
      setMetrics(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to retrieve analysis');
      setMetrics(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalysis(targetRoom);
  }, []);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchAnalysis(targetRoom);
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="rounded-2xl bg-slate-900/90 border border-slate-800 p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-mono">
                M3 — Communication Data Analysis
              </span>
              <span className="text-[11px] text-slate-400 font-mono hidden sm:inline">
                Calculated from persisted conversation messages
              </span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight flex items-center gap-2">
              <BarChart2 className="w-6 h-6 text-emerald-400" />
              <span>Dyadic Communication & Turn Dynamics</span>
            </h2>
            <p className="text-xs text-slate-400">
              Descriptive interaction metrics and turn latency statistics derived directly from stored database records.
            </p>
          </div>

          {/* Room Selector and Refresh */}
          <form onSubmit={handleSearch} className="flex items-center gap-2">
            <div className="relative">
              <input
                type="text"
                value={targetRoom}
                onChange={(e) => setTargetRoom(e.target.value.toUpperCase())}
                placeholder="Room (e.g. TEST001)"
                className="w-32 sm:w-40 px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono text-white uppercase placeholder-slate-500 focus:outline-none focus:border-emerald-500"
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-emerald-950/40 transition-all cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh Analysis</span>
            </button>
          </form>
        </div>

        {/* Non-Diagnostic Disclaimer */}
        <div className="p-3 rounded-xl bg-amber-950/30 border border-amber-800/40 text-amber-300 text-xs flex items-start gap-2.5">
          <Shield className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <strong>Analytical Research Specification:</strong> These metrics describe communication cadence, participation balance, and response latency. This system is an analytical pre-screening prototype and does <strong>not</strong> make clinical diagnoses or medical inferences.
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/50 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {metrics && (
        <div className="space-y-6">
          {/* Top Level Summary Cards (15 Metrics Grid) */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {/* 1. Total Messages */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">1. Total Messages</span>
              <p className="text-2xl font-bold text-white font-mono">{metrics.message_count}</p>
              <span className="text-[10px] text-slate-500 font-mono">Persisted turns</span>
            </div>

            {/* 2. Text Message Count */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">2. Text Messages</span>
              <p className="text-2xl font-bold text-emerald-400 font-mono">{metrics.text_message_count}</p>
              <span className="text-[10px] text-slate-500 font-mono">Modality 1 input</span>
            </div>

            {/* 3. Voice Message Count */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">3. Voice Messages</span>
              <p className="text-2xl font-bold text-indigo-400 font-mono">{metrics.voice_message_count}</p>
              <span className="text-[10px] text-slate-500 font-mono">Modality 2 input</span>
            </div>

            {/* 4. Average Length (Chars) */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">4. Avg Length (Chars)</span>
              <p className="text-2xl font-bold text-teal-400 font-mono">{metrics.average_message_length}</p>
              <span className="text-[10px] text-slate-500 font-mono">Characters / message</span>
            </div>

            {/* 5. Average Length (Words) */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">5. Avg Length (Words)</span>
              <p className="text-2xl font-bold text-teal-400 font-mono">{metrics.average_message_length_words}</p>
              <span className="text-[10px] text-slate-500 font-mono">Words / message</span>
            </div>

            {/* 6. Min Message Length */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">6. Min Message Length</span>
              <p className="text-2xl font-bold text-slate-200 font-mono">{metrics.min_message_length} chars</p>
              <span className="text-[10px] text-slate-500 font-mono">Lower bound</span>
            </div>

            {/* 7. Max Message Length */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">7. Max Message Length</span>
              <p className="text-2xl font-bold text-slate-200 font-mono">{metrics.max_message_length} chars</p>
              <span className="text-[10px] text-slate-500 font-mono">Upper bound</span>
            </div>

            {/* 8. Conversation Duration */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">8. Duration</span>
              <p className="text-2xl font-bold text-indigo-300 font-mono">{metrics.conversation_duration_seconds}s</p>
              <span className="text-[10px] text-slate-500 font-mono">First to last turn</span>
            </div>

            {/* 9. Average Response Time */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">9. Avg Response Time</span>
              <p className="text-2xl font-bold text-cyan-400 font-mono">
                {metrics.average_response_time_seconds !== null ? `${metrics.average_response_time_seconds}s` : 'N/A'}
              </p>
              <span className="text-[10px] text-slate-500 font-mono">Mean turn latency</span>
            </div>

            {/* 10. Min & Max Response Time */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">10/11. Min / Max Latency</span>
              <p className="text-xl font-bold text-cyan-300 font-mono">
                {metrics.min_response_time_seconds !== null ? `${metrics.min_response_time_seconds}s` : 'N/A'} /{' '}
                {metrics.max_response_time_seconds !== null ? `${metrics.max_response_time_seconds}s` : 'N/A'}
              </p>
              <span className="text-[10px] text-slate-500 font-mono">Latency bounds</span>
            </div>

            {/* 12/13. Gaps & Threshold */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">12/13. Gaps (≥{metrics.gap_threshold_seconds}s)</span>
              <p className="text-2xl font-bold text-amber-400 font-mono">{metrics.number_of_communication_gaps}</p>
              <span className="text-[10px] text-slate-500 font-mono">Inactivity episodes</span>
            </div>

            {/* Room Status */}
            <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="text-[11px] font-medium text-slate-400">Session Status</span>
              <p className="text-lg font-bold text-emerald-400 uppercase font-mono">{metrics.status}</p>
              <span className="text-[10px] text-slate-500 font-mono">Room: {metrics.room_id}</span>
            </div>
          </div>

          {/* Participant Breakdown Table */}
          <div className="rounded-2xl bg-slate-900/90 border border-slate-800 overflow-hidden shadow-xl">
            <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <User className="w-4 h-4 text-emerald-400" />
                <h3 className="font-bold text-white text-sm">Participant Contributions & Ratios</h3>
              </div>
              <span className="text-xs text-slate-400 font-mono">{metrics.participants.length} Participants</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-950/80 text-slate-400 font-mono">
                    <th className="p-3">Participant</th>
                    <th className="p-3">User ID</th>
                    <th className="p-3">Messages Sent</th>
                    <th className="p-3">Message Ratio</th>
                    <th className="p-3">Avg Chars</th>
                    <th className="p-3">Avg Words</th>
                    <th className="p-3">Total Words</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  {metrics.participant_breakdown.map((p) => (
                    <tr key={p.user_id} className="hover:bg-slate-800/40">
                      <td className="p-3 font-sans font-semibold text-white flex items-center gap-2">
                        <div className="w-6 h-6 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-xs font-bold font-mono">
                          {p.display_name.charAt(0)}
                        </div>
                        <span>{p.display_name}</span>
                      </td>
                      <td className="p-3 text-slate-400">{p.user_id}</td>
                      <td className="p-3 font-bold text-white">{p.message_count}</td>
                      <td className="p-3">
                        <div className="flex items-center gap-2">
                          <span className="text-emerald-400 font-bold">{Math.round(p.message_percentage * 100)}%</span>
                          <div className="w-16 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                            <div
                              className="h-full bg-emerald-500"
                              style={{ width: `${Math.round(p.message_percentage * 100)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td className="p-3">{p.average_message_length_chars}</td>
                      <td className="p-3">{p.average_message_length_words}</td>
                      <td className="p-3 text-slate-400">{p.total_words}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Academic Demonstration: Show Calculations Expandable Section */}
          <div className="rounded-2xl bg-slate-900/90 border border-slate-800 overflow-hidden shadow-xl">
            <button
              onClick={() => setShowCalculations(!showCalculations)}
              className="w-full px-5 py-4 flex items-center justify-between text-left hover:bg-slate-800/50 transition-colors cursor-pointer"
            >
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                  <Calculator className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-white text-sm sm:text-base">
                    Academic Step-by-Step Mathematical Calculations
                  </h3>
                  <p className="text-xs text-slate-400">
                    Transparent formula breakdown, input parameters, and substitutions for every metric.
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400">
                <span>{showCalculations ? 'Hide Calculations' : 'Show Calculations'}</span>
                {showCalculations ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </div>
            </button>

            {showCalculations && (
              <div className="p-5 border-t border-slate-800 bg-slate-950/60 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Calculation Card: Participation Ratios */}
                  {metrics.participant_breakdown.map((p) => (
                    <div key={`calc-${p.user_id}`} className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs">
                      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                        <span className="font-bold text-emerald-300 font-sans">{p.display_name} Message Ratio</span>
                        <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-mono text-[10px]">
                          COM-03
                        </span>
                      </div>
                      <div className="font-mono space-y-1 text-slate-300">
                        <p><strong className="text-slate-400">Formula:</strong> R_{p.display_name} = N_{p.display_name} / N_total</p>
                        <p><strong className="text-slate-400">Input:</strong> N_{p.display_name} = {p.message_count}, N_total = {metrics.message_count}</p>
                        <p><strong className="text-slate-400">Substitution:</strong> {p.message_count} / {metrics.message_count} = {p.message_percentage}</p>
                        <p className="text-emerald-400 font-bold"><strong className="text-slate-400">Result:</strong> {Math.round(p.message_percentage * 100)}% of conversation</p>
                      </div>
                    </div>
                  ))}

                  {/* Calculation Card: Average Message Length (Chars) */}
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <span className="font-bold text-teal-300 font-sans">Average Message Length (Characters)</span>
                      <span className="px-2 py-0.5 rounded bg-teal-500/10 text-teal-400 font-mono text-[10px]">
                        COM-02
                      </span>
                    </div>
                    <div className="font-mono space-y-1 text-slate-300">
                      <p><strong className="text-slate-400">Formula:</strong> L_avg = (Σ len(m_i)) / N_text</p>
                      <p><strong className="text-slate-400">Input:</strong> N_text = {metrics.text_message_count}, Bounds = [{metrics.min_message_length}, {metrics.max_message_length}]</p>
                      <p><strong className="text-slate-400">Substitution:</strong> Mean character count over {metrics.text_message_count} text messages</p>
                      <p className="text-teal-400 font-bold"><strong className="text-slate-400">Result:</strong> {metrics.average_message_length} characters / message</p>
                    </div>
                  </div>

                  {/* Calculation Card: Average Message Length (Words) */}
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <span className="font-bold text-teal-300 font-sans">Average Message Length (Words)</span>
                      <span className="px-2 py-0.5 rounded bg-teal-500/10 text-teal-400 font-mono text-[10px]">
                        COM-02
                      </span>
                    </div>
                    <div className="font-mono space-y-1 text-slate-300">
                      <p><strong className="text-slate-400">Formula:</strong> W_avg = (Σ word_count(m_i)) / N_text</p>
                      <p><strong className="text-slate-400">Input:</strong> N_text = {metrics.text_message_count}</p>
                      <p><strong className="text-slate-400">Substitution:</strong> Whitespace-tokenized count / {metrics.text_message_count}</p>
                      <p className="text-teal-400 font-bold"><strong className="text-slate-400">Result:</strong> {metrics.average_message_length_words} words / message</p>
                    </div>
                  </div>

                  {/* Calculation Card: Conversation Duration */}
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <span className="font-bold text-indigo-300 font-sans">Conversation Duration</span>
                      <span className="px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 font-mono text-[10px]">
                        COM-02
                      </span>
                    </div>
                    <div className="font-mono space-y-1 text-slate-300">
                      <p><strong className="text-slate-400">Formula:</strong> D = timestamp(m_last) - timestamp(m_first)</p>
                      <p><strong className="text-slate-400">Input:</strong> Earliest and latest UTC message timestamps</p>
                      <p><strong className="text-slate-400">Substitution:</strong> Total elapsed time in seconds</p>
                      <p className="text-indigo-400 font-bold"><strong className="text-slate-400">Result:</strong> {metrics.conversation_duration_seconds} seconds</p>
                    </div>
                  </div>

                  {/* Calculation Card: Turn Latency (Response Time) */}
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <span className="font-bold text-cyan-300 font-sans">Turn-Taking Response Latency</span>
                      <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-mono text-[10px]">
                        COM-01
                      </span>
                    </div>
                    <div className="font-mono space-y-1 text-slate-300">
                      <p><strong className="text-slate-400">Formula:</strong> T_resp = (1 / K) Σ (t_{`{k+1}`} - t_k) [for sender transition]</p>
                      <p><strong className="text-slate-400">Input:</strong> Min = {metrics.min_response_time_seconds ?? 'N/A'}s, Max = {metrics.max_response_time_seconds ?? 'N/A'}s</p>
                      <p><strong className="text-slate-400">Substitution:</strong> Latency between alternating speaker transitions</p>
                      <p className="text-cyan-400 font-bold"><strong className="text-slate-400">Result:</strong> {metrics.average_response_time_seconds !== null ? `${metrics.average_response_time_seconds}s avg latency` : 'No speaker transitions'}</p>
                    </div>
                  </div>

                  {/* Calculation Card: Communication Gaps */}
                  <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <span className="font-bold text-amber-300 font-sans">Inactivity Communication Gaps</span>
                      <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 font-mono text-[10px]">
                        COM-02
                      </span>
                    </div>
                    <div className="font-mono space-y-1 text-slate-300">
                      <p><strong className="text-slate-400">Formula:</strong> N_gaps = Σ 𝟙(Δt_k ≥ τ) [where τ = {metrics.gap_threshold_seconds}s]</p>
                      <p><strong className="text-slate-400">Input:</strong> Inactivity threshold τ = {metrics.gap_threshold_seconds} seconds</p>
                      <p><strong className="text-slate-400">Substitution:</strong> Count of transitions where latency ≥ {metrics.gap_threshold_seconds}s</p>
                      <p className="text-amber-400 font-bold"><strong className="text-slate-400">Result:</strong> {metrics.number_of_communication_gaps} detected gap(s)</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
