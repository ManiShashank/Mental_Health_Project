import React, { useState, useEffect, useRef } from 'react';
import {
  MessageSquare,
  Download,
  Info,
  Send,
  Users,
  LogOut,
  Wifi,
  WifiOff,
  RefreshCw,
  AlertCircle,
  Clock,
  Shield,
  Hash,
  UserCheck,
  CornerDownLeft,
  BarChart2,
  ChevronDown,
  ChevronUp,
  Volume2,
  FileText,
  Sparkles,
  Loader2,
  Activity,
  Waves,
  BookOpen,
  Smile,
  Layers,
  GitMerge,
  Sliders,
  CheckCircle2,
  XCircle,
  TrendingUp,
  Compass
} from 'lucide-react';
import { ChatWebSocketClient, ConnectionStatus } from '../services/websocket';
import {
  ChatMessage,
  ActiveParticipant,
  AcousticFeatureData,
  NLPFeatureData,
  MultimodalFeatureData,
  TemporalTrendData,
  PersonalBaselineData,
  AnalyticalReportData
} from '../types';
import { CommunicationAnalysisView } from '../components/CommunicationAnalysisView';
import { VoiceRecorder } from '../components/VoiceRecorder';
import {
  transcribeVoiceMessage,
  analyzeAcousticFeatures,
  getAcousticFeatures,
  analyzeNLPFeatures,
  getNLPFeatures,
  fuseMultimodalFeatures,
  getMultimodalFeatures,
  evaluateMessageTemporal,
  getMessageTemporal,
  getUserBaseline,
  generateAnalyticalReport,
  getLatestAnalyticalReport,
  exportAnalyticalReport
} from '../services/api';

export const ChatRoom: React.FC = () => {
  // Lobby State
  const [inRoom, setInRoom] = useState<boolean>(false);
  const [userName, setUserName] = useState<string>(() => localStorage.getItem('chat_user_name') || '');
  const [roomId, setRoomId] = useState<string>(() => localStorage.getItem('chat_room_id') || 'TEST001');
  const [userId] = useState<string>(() => {
    let stored = localStorage.getItem('chat_user_id');
    if (!stored) {
      stored = `PID-${Math.floor(1000 + Math.random() * 9000)}`;
      localStorage.setItem('chat_user_id', stored);
    }
    return stored;
  });

  // Milestone 10: Analytical Report State
  const [isReportModalOpen, setIsReportModalOpen] = useState<boolean>(false);
  const [currentReport, setCurrentReport] = useState<AnalyticalReportData | null>(null);
  const [isLoadingReport, setIsLoadingReport] = useState<boolean>(false);
  const [reportError, setReportError] = useState<string | null>(null);

  // Active Session State
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [participants, setParticipants] = useState<ActiveParticipant[]>([]);
  const [inputText, setInputText] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Milestone 3: Communication Analytics Panel State
  const [showAnalytics, setShowAnalytics] = useState<boolean>(false);

  // Milestone 5: Speech-to-Text Transcribing Map
  const [transcribingMap, setTranscribingMap] = useState<Record<string, boolean>>({});

  // Milestone 6: Acoustic Analysis State Maps
  const [analyzingAcousticMap, setAnalyzingAcousticMap] = useState<Record<string, boolean>>({});
  const [acousticDataMap, setAcousticDataMap] = useState<Record<string, AcousticFeatureData>>({});
  const [expandedAcousticMap, setExpandedAcousticMap] = useState<Record<string, boolean>>({});

  // Milestone 7: NLP & Linguistic Feature Extraction State Maps
  const [analyzingNLPMap, setAnalyzingNLPMap] = useState<Record<string, boolean>>({});
  const [nlpDataMap, setNLPDataMap] = useState<Record<string, NLPFeatureData>>({});
  const [expandedNLPMap, setExpandedNLPMap] = useState<Record<string, boolean>>({});

  // Milestone 8: Multimodal Feature Fusion State Maps
  const [fusingMultimodalMap, setFusingMultimodalMap] = useState<Record<string, boolean>>({});
  const [multimodalDataMap, setMultimodalDataMap] = useState<Record<string, MultimodalFeatureData>>({});
  const [expandedMultimodalMap, setExpandedMultimodalMap] = useState<Record<string, boolean>>({});

  // Milestone 9: Personal Baseline & Temporal Trend State Maps
  const [analyzingTemporalMap, setAnalyzingTemporalMap] = useState<Record<string, boolean>>({});
  const [temporalDataMap, setTemporalDataMap] = useState<Record<string, TemporalTrendData>>({});
  const [expandedTemporalMap, setExpandedTemporalMap] = useState<Record<string, boolean>>({});
  const [userBaseline, setUserBaseline] = useState<PersonalBaselineData | null>(null);
  const [, setLoadingBaseline] = useState<boolean>(false);


  // References
  const clientRef = useRef<ChatWebSocketClient | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const handleTranscribe = async (messageId: string) => {
    setTranscribingMap((prev) => ({ ...prev, [messageId]: true }));
    try {
      const res = await transcribeVoiceMessage(messageId);
      setMessages((prev) =>
        prev.map((m) =>
          m.message_id === messageId
            ? {
                ...m,
                transcript_text: res.transcript_text,
                transcription_status: 'completed',
                transcription_model: res.transcription_model,
                transcription_language: res.transcription_language,
                transcribed_at: res.transcribed_at,
              }
            : m
        )
      );
    } catch (err: any) {
      console.error('Transcription error:', err);
      setMessages((prev) =>
        prev.map((m) =>
          m.message_id === messageId
            ? {
                ...m,
                transcription_status: 'failed',
                transcription_error: err.response?.data?.detail || err.message || 'Transcription failed',
              }
            : m
        )
      );
    } finally {
      setTranscribingMap((prev) => ({ ...prev, [messageId]: false }));
    }
  };

  const handleAnalyzeAcoustic = async (messageId: string) => {
    setAnalyzingAcousticMap((prev) => ({ ...prev, [messageId]: true }));
    try {
      const res = await analyzeAcousticFeatures(messageId);
      setAcousticDataMap((prev) => ({ ...prev, [messageId]: res }));
      setExpandedAcousticMap((prev) => ({ ...prev, [messageId]: true }));
      setMessages((prev) =>
        prev.map((m) =>
          m.message_id === messageId
            ? { ...m, has_acoustic_features: true }
            : m
        )
      );
    } catch (err: any) {
      console.error('Acoustic analysis error:', err);
      alert(err.response?.data?.detail || err.message || 'Acoustic extraction failed');
    } finally {
      setAnalyzingAcousticMap((prev) => ({ ...prev, [messageId]: false }));
    }
  };

  const toggleAcousticDrawer = async (messageId: string) => {
    const isExpanded = !!expandedAcousticMap[messageId];
    if (!isExpanded && !acousticDataMap[messageId]) {
      try {
        const data = await getAcousticFeatures(messageId);
        setAcousticDataMap((prev) => ({ ...prev, [messageId]: data }));
      } catch {
        // If not yet analyzed, user can click Analyze
      }
    }
    setExpandedAcousticMap((prev) => ({ ...prev, [messageId]: !isExpanded }));
  };

  const handleAnalyzeNLP = async (messageId: string) => {
    setAnalyzingNLPMap((prev) => ({ ...prev, [messageId]: true }));
    try {
      const res = await analyzeNLPFeatures(messageId);
      setNLPDataMap((prev) => ({ ...prev, [messageId]: res }));
      setExpandedNLPMap((prev) => ({ ...prev, [messageId]: true }));
      setMessages((prev) =>
        prev.map((m) =>
          m.message_id === messageId
            ? { ...m, has_nlp_features: true }
            : m
        )
      );
    } catch (err: any) {
      console.error('NLP analysis error:', err);
      alert(err.response?.data?.detail || err.message || 'NLP extraction failed');
    } finally {
      setAnalyzingNLPMap((prev) => ({ ...prev, [messageId]: false }));
    }
  };

  const toggleNLPDrawer = async (messageId: string) => {
    const isExpanded = !!expandedNLPMap[messageId];
    if (!isExpanded && !nlpDataMap[messageId]) {
      try {
        const data = await getNLPFeatures(messageId);
        setNLPDataMap((prev) => ({ ...prev, [messageId]: data }));
      } catch {
        // If not yet analyzed, user can click Analyze
      }
    }
    setExpandedNLPMap((prev) => ({ ...prev, [messageId]: !isExpanded }));
  };

const handleFuseMultimodal = async (messageId: string) => {
    setFusingMultimodalMap((prev) => ({ ...prev, [messageId]: true }));
    try {
      const res = await fuseMultimodalFeatures(messageId);
      setMultimodalDataMap((prev) => ({ ...prev, [messageId]: res }));
      setExpandedMultimodalMap((prev) => ({ ...prev, [messageId]: true }));
      setMessages((prev) =>
        prev.map((m) =>
          m.message_id === messageId
            ? { ...m, has_multimodal_features: true }
            : m
        )
      );
    } catch (err: any) {
      console.error('Multimodal fusion error:', err);
      alert(err.response?.data?.detail || err.message || 'Multimodal fusion failed');
    } finally {
      setFusingMultimodalMap((prev) => ({ ...prev, [messageId]: false }));
    }
  };

  const toggleMultimodalDrawer = async (messageId: string) => {
    const isExpanded = !!expandedMultimodalMap[messageId];
    if (!isExpanded && !multimodalDataMap[messageId]) {
      try {
        const data = await getMultimodalFeatures(messageId);
        setMultimodalDataMap((prev) => ({ ...prev, [messageId]: data }));
      } catch {
        // Not yet fused, can click Fuse
      }
    }
    setExpandedMultimodalMap((prev) => ({ ...prev, [messageId]: !isExpanded }));
  };

  const fetchBaseline = async () => {
    try {
      setLoadingBaseline(true);
      const data = await getUserBaseline(userId);
      setUserBaseline(data);
    } catch (e) {
      console.warn("Failed to fetch user baseline:", e);
    } finally {
      setLoadingBaseline(false);
    }
  };

  const handleEvaluateTemporal = async (messageId: string) => {
    setAnalyzingTemporalMap((prev) => ({ ...prev, [messageId]: true }));
    try {
      const res = await evaluateMessageTemporal(messageId);
      setTemporalDataMap((prev) => ({ ...prev, [messageId]: res }));
      setExpandedTemporalMap((prev) => ({ ...prev, [messageId]: true }));
      setMessages((prev) =>
        prev.map((m) =>
          m.message_id === messageId
            ? { ...m, has_temporal_trends: true }
            : m
        )
      );
      fetchBaseline();
    } catch (err: any) {
      console.error('Temporal trend evaluation error:', err);
      alert(err.response?.data?.detail || err.message || 'Temporal trend evaluation failed');
    } finally {
      setAnalyzingTemporalMap((prev) => ({ ...prev, [messageId]: false }));
    }
  };

  const toggleTemporalDrawer = async (messageId: string) => {
    const isExpanded = !!expandedTemporalMap[messageId];
    if (!isExpanded && !temporalDataMap[messageId]) {
      try {
        const data = await getMessageTemporal(messageId);
        setTemporalDataMap((prev) => ({ ...prev, [messageId]: data }));
      } catch {
        // Not yet evaluated
      }
    }
    setExpandedTemporalMap((prev) => ({ ...prev, [messageId]: !isExpanded }));
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (inRoom) {
      scrollToBottom();
      fetchBaseline();
    }
  }, [messages, inRoom]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (clientRef.current) {
        clientRef.current.disconnect();
      }
    };
  }, []);

  const handleJoinRoom = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const cleanName = userName.trim();
    const cleanRoom = roomId.trim();

    if (!cleanName) {
      setErrorMessage('Please enter your display name.');
      return;
    }
    if (!cleanRoom) {
      setErrorMessage('Please enter a room identifier.');
      return;
    }

    localStorage.setItem('chat_user_name', cleanName);
    localStorage.setItem('chat_room_id', cleanRoom);
    setErrorMessage(null);
    setMessages([]);
    setParticipants([]);

    // Initialize WebSocket client
    const client = new ChatWebSocketClient(cleanRoom, userId, cleanName, {
      onStatusChange: (status) => setConnectionStatus(status),
      onHistory: (historyMsgs, roomParticipants) => {
        setMessages(historyMsgs);
        setParticipants(roomParticipants);
      },
      onMessage: (msg) => {
        setMessages((prev) => [...prev, msg]);
      },
      onTranscriptionCompleted: (data) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.message_id === data.message_id
              ? {
                  ...m,
                  transcript_text: data.transcript_text,
                  transcription_status: 'completed',
                  transcription_model: data.transcription_model,
                  transcription_language: data.transcription_language,
                  transcribed_at: data.transcribed_at,
                }
              : m
          )
        );
      },
      onAcousticAnalysisCompleted: (data) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.message_id === data.message_id
              ? { ...m, has_acoustic_features: true }
              : m
          )
        );
      },
      onNLPAnalysisCompleted: (data) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.message_id === data.message_id
              ? { ...m, has_nlp_features: true }
              : m
          )
        );
      },
      onMultimodalFusionCompleted: (data) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.message_id === data.message_id
              ? { ...m, has_multimodal_features: true }
              : m
          )
        );
      },
      onPersonalBaselineUpdated: (data) => {
        if (data.user_id === userId) {
          fetchBaseline();
        }
      },
      onAnalyticalReportGenerated: (data) => {
          if (data.user_id === userId) {
            getLatestAnalyticalReport(userId, roomId).then(setCurrentReport).catch(console.warn);
          }
        },
        onTemporalDeviationComputed: (data) => {
        setMessages((prev) =>
          prev.map((m) =>
            m.message_id === data.message_id
              ? { ...m, has_temporal_trends: true }
              : m
          )
        );
      },
      onUserJoined: (data) => {
        setParticipants(data.participants);
        setMessages((prev) => [
          ...prev,
          {
            message_id: `sys-${Date.now()}`,
            conversation_id: cleanRoom,
            room_id: cleanRoom,
            sender_id: 'SYSTEM',
            sender_name: 'System',
            message_text: data.system_message,
            timestamp: new Date().toISOString(),
            message_type: 'system',
          },
        ]);
      },
      onUserLeft: (data) => {
        setParticipants(data.participants);
        setMessages((prev) => [
          ...prev,
          {
            message_id: `sys-${Date.now()}`,
            conversation_id: cleanRoom,
            room_id: cleanRoom,
            sender_id: 'SYSTEM',
            sender_name: 'System',
            message_text: data.system_message,
            timestamp: new Date().toISOString(),
            message_type: 'system',
          },
        ]);
      },
      onError: (err) => {
        setErrorMessage(err);
      },
    });

    clientRef.current = client;
    client.connect();
    setInRoom(true);
  };

  const handleLeaveRoom = () => {
    if (clientRef.current) {
      clientRef.current.disconnect();
      clientRef.current = null;
    }
    setInRoom(false);
    setConnectionStatus('disconnected');
    setMessages([]);
    setShowAnalytics(false);
  };

  const handleSendMessage = () => {
    const trimmed = inputText.trim();
    if (!trimmed) {
      setErrorMessage('Cannot send empty message.');
      return;
    }

    if (!clientRef.current || connectionStatus !== 'connected') {
      setErrorMessage('Cannot send message: Not connected to chat server.');
      return;
    }

    setErrorMessage(null);
    const success = clientRef.current.sendMessage(trimmed);
    if (success) {
      setInputText('');
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Milestone 10: Analytical Report Handlers
  const handleOpenReport = async () => {
    setIsReportModalOpen(true);
    setIsLoadingReport(true);
    setReportError(null);
    try {
      const rep = await generateAnalyticalReport(userId, { room_id: roomId });
      setCurrentReport(rep);
    } catch (err: any) {
      console.warn('Failed to generate report, falling back to latest cached:', err);
      try {
        const cached = await getLatestAnalyticalReport(userId, roomId);
        setCurrentReport(cached);
      } catch (cachedErr: any) {
        setReportError(
          err?.response?.data?.detail ||
          cachedErr?.response?.data?.detail ||
          'Failed to generate or retrieve analytical report. Ensure at least one observation exists.'
        );
      }
    } finally {
      setIsLoadingReport(false);
    }
  };

  const handleExport = async (format: 'json' | 'markdown' | 'html') => {
    if (!currentReport) return;
    try {
      const data = await exportAnalyticalReport(currentReport.metadata.report_id, format);
      let blob: Blob;
      let filename: string;
      if (format === 'json') {
        blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        filename = `analytical_report_${userId}_${currentReport.metadata.report_id.slice(0, 8)}.json`;
      } else if (format === 'markdown') {
        blob = new Blob([data], { type: 'text/markdown' });
        filename = `analytical_report_${userId}_${currentReport.metadata.report_id.slice(0, 8)}.md`;
      } else {
        blob = new Blob([data], { type: 'text/html' });
        filename = `analytical_report_${userId}_${currentReport.metadata.report_id.slice(0, 8)}.html`;
      }
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error('Export failed:', err);
      alert('Failed to export analytical report.');
    }
  };

  // Format UTC Timestamp
  const formatTime = (ts: string) => {
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return ts;
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-12">
      {/* Non-Diagnostic Advisory */}
      <div className="rounded-xl bg-slate-900/60 border border-slate-800 p-3 text-xs text-slate-400 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Shield className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          <span>
            <strong>M6 — Voice Acoustics, Speech-to-Text & Multimodal Chat:</strong> Real-time dyadic communication with voice recording, Speech-to-Text transcription, and descriptive acoustic signal extraction ($F_0$, RMS Energy, Speech Rate, Pauses, MFCCs, Spectral Descriptors).
          </span>
        </div>
        <span className="font-mono text-[11px] text-slate-500 hidden sm:inline">Participant ID: {userId}</span>
      </div>

      {/* LOBBY VIEW */}
      {!inRoom ? (
        <div className="max-w-lg mx-auto py-8">
          <div className="rounded-2xl bg-slate-900/90 border border-slate-800 p-6 sm:p-8 space-y-6 shadow-2xl">
            <div className="text-center space-y-2">
              <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-center mx-auto shadow-lg shadow-emerald-950/40">
                <MessageSquare className="w-6 h-6" />
              </div>
              <h2 className="text-2xl font-bold text-white tracking-tight">Join Dyadic Chat Room</h2>
              <p className="text-xs text-slate-400">
                Enter your participant name and target room code to start real-time messaging.
              </p>
            </div>

            {errorMessage && (
              <div className="p-3 rounded-xl bg-rose-950/50 border border-rose-800/60 text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-400" />
                <span>{errorMessage}</span>
              </div>
            )}

            <form onSubmit={handleJoinRoom} className="space-y-4">
              <div className="space-y-1.5">
                <label className="block text-xs font-semibold text-slate-300">Display Name / Alias</label>
                <div className="relative">
                  <input
                    type="text"
                    required
                    maxLength={64}
                    placeholder="e.g., Alice or Participant A"
                    value={userName}
                    onChange={(e) => setUserName(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
                  />
                  <UserCheck className="w-4 h-4 text-slate-500 absolute right-3 top-3" />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="block text-xs font-semibold text-slate-300">Room Identifier</label>
                <div className="relative">
                  <input
                    type="text"
                    required
                    maxLength={32}
                    placeholder="e.g., TEST001"
                    value={roomId}
                    onChange={(e) => setRoomId(e.target.value.toUpperCase())}
                    className="w-full px-4 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-sm text-slate-100 font-mono placeholder-slate-500 focus:outline-none focus:border-emerald-500 uppercase"
                  />
                  <Hash className="w-4 h-4 text-slate-500 absolute right-3 top-3" />
                </div>
                <p className="text-[11px] text-slate-500">
                  Tip: Use the same room code (e.g. <code className="text-emerald-400">TEST001</code>) across two browser tabs to test multi-user interaction.
                </p>
              </div>

              <button
                type="submit"
                className="w-full py-3 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-sm shadow-lg shadow-emerald-950/50 transition-all flex items-center justify-center gap-2 cursor-pointer"
              >
                <MessageSquare className="w-4 h-4" />
                <span>Enter Room</span>
              </button>
            </form>
          </div>
        </div>
      ) : (
        /* ACTIVE CHAT ROOM VIEW */
        <div className="space-y-4">
          <div className="rounded-2xl bg-slate-900/90 border border-slate-800 overflow-hidden shadow-2xl flex flex-col h-[75vh]">
            {/* Room Header */}
            <div className="bg-slate-950/90 border-b border-slate-800 px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                  <Hash className="w-4 h-4" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-white text-sm sm:text-base tracking-tight font-mono">
                      Room {roomId}
                    </h3>
                    {/* Connection Status Badge */}
                    <span
                      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${
                        connectionStatus === 'connected'
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : connectionStatus === 'reconnecting'
                          ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                          : connectionStatus === 'connecting'
                          ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30'
                          : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                      }`}
                    >
                      {connectionStatus === 'connected' ? (
                        <>
                          <Wifi className="w-3 h-3 text-emerald-400" />
                          <span>Connected</span>
                        </>
                      ) : connectionStatus === 'reconnecting' ? (
                        <>
                          <RefreshCw className="w-3 h-3 animate-spin text-amber-400" />
                          <span>Reconnecting</span>
                        </>
                      ) : (
                        <>
                          <WifiOff className="w-3 h-3 text-rose-400" />
                          <span>Disconnected</span>
                        </>
                      )}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Logged in as <strong className="text-slate-200">{userName}</strong> ({userId})
                  </p>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center gap-2 sm:gap-3">
                {/* Personal Baseline Status Pill */}
                <div
                  className={`hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-mono transition-all ${
                    userBaseline?.is_baseline_established
                      ? 'bg-emerald-950/50 text-emerald-300 border-emerald-800/80'
                      : 'bg-amber-950/50 text-amber-300 border-amber-800/80'
                  }`}
                  title={
                    userBaseline?.is_baseline_established
                      ? 'Idiographic baseline established (N >= 3 prior observations)'
                      : 'Collecting historical observations to calibrate personal normative baseline'
                  }
                >
                  <Compass className={`w-3.5 h-3.5 ${userBaseline?.is_baseline_established ? 'text-emerald-400' : 'text-amber-400'}`} />
                  {userBaseline?.is_baseline_established ? (
                    <span className="font-semibold">Baseline Active (N={userBaseline.sample_count})</span>
                  ) : (
                    <span>Baseline Calibrating ({userBaseline?.sample_count || 0}/3)</span>
                  )}
                </div>

                                {/* Milestone 10: Analytical Report Button */}
                <button
                  onClick={handleOpenReport}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/40 text-xs font-semibold transition-all cursor-pointer shadow-sm shadow-indigo-950/40"
                  title="Generate Longitudinal Analytical Report (Milestone 10)"
                >
                  <FileText className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Analytical Report</span>
                </button>

                {/* Live Metrics Toggle Button */}
                <button
                  onClick={() => setShowAnalytics(!showAnalytics)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all cursor-pointer ${
                    showAnalytics
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
                  }`}
                >
                  <BarChart2 className="w-3.5 h-3.5" />
                  <span>Metrics</span>
                  {showAnalytics ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>

                <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300">
                  <Users className="w-3.5 h-3.5 text-indigo-400" />
                  <span className="font-semibold">{participants.length} Active</span>
                </div>

                <button
                  onClick={handleLeaveRoom}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-rose-950/50 hover:text-rose-300 hover:border-rose-800 text-xs font-semibold text-slate-300 border border-slate-700 transition-all cursor-pointer"
                >
                  <LogOut className="w-3.5 h-3.5" />
                  <span>Leave</span>
                </button>
              </div>
            </div>

            {/* Error Banner */}
            {errorMessage && (
              <div className="bg-rose-950/40 border-b border-rose-800/40 px-4 py-2 text-xs text-rose-300 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
                  <span>{errorMessage}</span>
                </div>
                <button
                  onClick={() => setErrorMessage(null)}
                  className="text-slate-400 hover:text-white text-xs font-bold"
                >
                  ✕
                </button>
              </div>
            )}

            {/* Messages Scroll Area */}
            <div className="flex-1 p-4 sm:p-6 overflow-y-auto space-y-4 bg-slate-950/40">
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center text-slate-500 space-y-2">
                  <MessageSquare className="w-8 h-8 opacity-40 text-slate-400" />
                  <p className="text-sm font-medium">No messages in Room {roomId} yet.</p>
                  <p className="text-xs max-w-sm">
                    Send a message below or open another tab in room <strong>{roomId}</strong> to test live interaction.
                  </p>
                </div>
              ) : (
                messages.map((msg) => {
                  if (msg.message_type === 'system') {
                    return (
                      <div key={msg.message_id} className="flex justify-center my-2">
                        <span className="px-3 py-1 rounded-full text-[11px] font-mono bg-slate-900 border border-slate-800 text-slate-400">
                          {msg.message_text}
                        </span>
                      </div>
                    );
                  }

                  const isSelf = msg.sender_id === userId;

                  return (
                    <div
                      key={msg.message_id}
                      className={`flex flex-col ${isSelf ? 'items-end' : 'items-start'}`}
                    >
                      <div className="flex items-center gap-2 mb-1 px-1 text-[11px] text-slate-400 font-mono">
                        <span className="font-semibold text-slate-200">
                          {isSelf ? `You (${userName})` : msg.sender_name || msg.sender_id}
                        </span>
                        <span>•</span>
                        <span className="flex items-center gap-1 text-slate-500">
                          <Clock className="w-2.5 h-2.5" />
                          {formatTime(msg.timestamp)}
                        </span>
                      </div>

                      {msg.message_type === 'voice' ? (
                        <div
                          className={`max-w-[85%] sm:max-w-md rounded-2xl p-3 space-y-2 shadow-lg ${
                            isSelf
                              ? 'bg-emerald-950/80 text-white rounded-br-none border border-emerald-500/50'
                              : 'bg-slate-800/90 text-slate-100 rounded-bl-none border border-slate-700'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2 text-xs">
                            <div className="flex items-center gap-1.5 font-semibold text-emerald-400">
                              <Volume2 className="w-4 h-4" />
                              <span>Voice Recording</span>
                            </div>
                            {msg.audio_duration_seconds !== undefined && msg.audio_duration_seconds !== null && (
                              <span className="px-2 py-0.5 rounded-full bg-slate-900/80 border border-slate-700 font-mono text-[10px] text-slate-300">
                                {msg.audio_duration_seconds}s
                              </span>
                            )}
                          </div>

                          {msg.audio_url ? (
                            <audio
                              controls
                              src={msg.audio_url}
                              className="w-full h-8 rounded-lg mt-1"
                              preload="metadata"
                            />
                          ) : (
                            <p className="text-xs text-rose-300 italic">Audio recording reference missing.</p>
                          )}

                          {/* Milestone 5: Speech-to-Text Transcript Section */}
                          {msg.transcript_text ? (
                            <div className="pt-2 border-t border-slate-700/60 space-y-1.5 text-xs">
                              <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                                <span className="flex items-center gap-1 font-semibold text-emerald-400">
                                  <FileText className="w-3 h-3" />
                                  Transcript ({msg.transcription_model || 'Whisper'}):
                                </span>
                                {msg.transcription_language && (
                                  <span className="px-1.5 py-0.5 rounded bg-slate-900 text-slate-300 font-mono text-[9px] border border-slate-800">
                                    {msg.transcription_language.toUpperCase()}
                                  </span>
                                )}
                              </div>
                              <p className="italic text-slate-200 bg-slate-900/80 p-2.5 rounded-xl border border-slate-800 leading-relaxed font-sans shadow-inner">
                                "{msg.transcript_text}"
                              </p>
                            </div>
                          ) : transcribingMap[msg.message_id] || msg.transcription_status === 'processing' ? (
                            <div className="pt-2 border-t border-slate-700/60 flex items-center gap-2 text-xs text-indigo-300 animate-pulse font-mono">
                              <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                              <span>Transcribing speech to text...</span>
                            </div>
                          ) : msg.transcription_status === 'failed' ? (
                            <div className="pt-2 border-t border-slate-700/60 flex items-center justify-between gap-2 text-xs text-rose-300">
                              <span className="text-[11px] font-mono">Transcription failed</span>
                              <button
                                onClick={() => handleTranscribe(msg.message_id)}
                                className="px-2 py-0.5 rounded bg-rose-900/60 hover:bg-rose-800 text-[10px] text-white cursor-pointer transition-colors"
                              >
                                Retry
                              </button>
                            </div>
                          ) : (
                            <div className="pt-2 border-t border-slate-700/60 flex items-center justify-between gap-2">
                              <span className="text-[10px] text-slate-400 font-mono">Audio transcript</span>
                              <button
                                onClick={() => handleTranscribe(msg.message_id)}
                                className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-emerald-400 hover:text-emerald-300 text-[11px] font-semibold border border-slate-700 flex items-center gap-1.5 cursor-pointer shadow transition-all"
                                title="Transcribe speech to text"
                              >
                                <Sparkles className="w-3 h-3" />
                                <span>Transcribe</span>
                              </button>
                            </div>
                          )}

                          {/* Milestone 6: Acoustic Feature Extraction Section */}
                          <div className="pt-2 border-t border-slate-700/60 space-y-2 text-xs">
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-[10px] text-slate-400 font-mono flex items-center gap-1">
                                <Activity className="w-3 h-3 text-cyan-400" />
                                Acoustic Descriptors (M6)
                              </span>
                              
                              <div className="flex items-center gap-1.5">
                                {analyzingAcousticMap[msg.message_id] ? (
                                  <span className="flex items-center gap-1 text-[10px] text-cyan-300 font-mono animate-pulse">
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                    Extracting...
                                  </span>
                                ) : acousticDataMap[msg.message_id] || msg.has_acoustic_features ? (
                                  <button
                                    onClick={() => toggleAcousticDrawer(msg.message_id)}
                                    className="px-2 py-0.5 rounded bg-cyan-950/70 hover:bg-cyan-900/80 text-cyan-300 text-[10px] font-mono border border-cyan-800/60 flex items-center gap-1 cursor-pointer transition-colors"
                                  >
                                    <Waves className="w-2.5 h-2.5" />
                                    {expandedAcousticMap[msg.message_id] ? 'Hide 59-d' : 'View 59-d'}
                                  </button>
                                ) : (
                                  <button
                                    onClick={() => handleAnalyzeAcoustic(msg.message_id)}
                                    className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300 text-[10px] font-mono border border-slate-700 flex items-center gap-1 cursor-pointer transition-colors"
                                    title="Extract pitch, energy, speech rate, pauses, MFCCs, and spectral descriptors"
                                  >
                                    <Activity className="w-2.5 h-2.5" />
                                    Analyze
                                  </button>
                                )}
                              </div>
                            </div>

                            {/* Collapsible Acoustic Breakdown */}
                            {expandedAcousticMap[msg.message_id] && acousticDataMap[msg.message_id] && (
                              <div className="bg-slate-900/95 border border-cyan-900/50 rounded-xl p-3 space-y-2.5 text-[11px] shadow-xl">
                                <div className="flex items-center justify-between text-[9px] text-slate-400 font-mono pb-1 border-b border-slate-800">
                                  <span>Engine: <strong className="text-slate-200">{acousticDataMap[msg.message_id].engine}</strong></span>
                                  <span className="px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-bold">
                                    {acousticDataMap[msg.message_id].feature_vector_version} (59 dims)
                                  </span>
                                </div>

                                {/* Feature Groups Grid */}
                                <div className="grid grid-cols-2 gap-2 text-[10px]">
                                  {/* Pitch (F0) */}
                                  <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                    <div className="font-semibold text-cyan-400 flex items-center gap-1">
                                      <span>Pitch (F₀)</span>
                                    </div>
                                    <p className="text-slate-300">Mean: <strong className="font-mono text-white">{acousticDataMap[msg.message_id].pitch_features.f0_mean_hz ? `${acousticDataMap[msg.message_id].pitch_features.f0_mean_hz?.toFixed(1)} Hz` : 'Unvoiced / N/A'}</strong></p>
                                    <p className="text-slate-400">Range: {acousticDataMap[msg.message_id].pitch_features.f0_range_hz ? `${acousticDataMap[msg.message_id].pitch_features.f0_range_hz?.toFixed(1)} Hz` : 'N/A'}</p>
                                    <p className="text-slate-400">Voicing: {(acousticDataMap[msg.message_id].pitch_features.voicing_fraction * 100).toFixed(1)}%</p>
                                    <p className="text-slate-400">Jitter: {acousticDataMap[msg.message_id].pitch_features.f0_jitter_local !== null && acousticDataMap[msg.message_id].pitch_features.f0_jitter_local !== undefined ? acousticDataMap[msg.message_id].pitch_features.f0_jitter_local?.toFixed(4) : 'N/A'}</p>
                                  </div>

                                  {/* Energy & Shimmer */}
                                  <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                    <div className="font-semibold text-emerald-400 flex items-center gap-1">
                                      <span>RMS Energy</span>
                                    </div>
                                    <p className="text-slate-300">Mean: <strong className="font-mono text-white">{acousticDataMap[msg.message_id].energy_features.rms_mean.toFixed(4)}</strong></p>
                                    <p className="text-slate-400">Peak: {acousticDataMap[msg.message_id].energy_features.rms_max.toFixed(4)}</p>
                                    <p className="text-slate-400">Range: {acousticDataMap[msg.message_id].energy_features.rms_energy_range.toFixed(4)}</p>
                                    <p className="text-slate-400">Shimmer: {acousticDataMap[msg.message_id].energy_features.shimmer_local !== null && acousticDataMap[msg.message_id].energy_features.shimmer_local !== undefined ? acousticDataMap[msg.message_id].energy_features.shimmer_local?.toFixed(4) : 'N/A'}</p>
                                  </div>

                                  {/* Speech Rate & Pauses */}
                                  <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                    <div className="font-semibold text-indigo-400 flex items-center gap-1">
                                      <span>Tempo & Pauses</span>
                                    </div>
                                    <p className="text-slate-300">Syllables (Est): <strong className="font-mono text-white">{acousticDataMap[msg.message_id].speech_rate_features.estimated_syllable_count}</strong></p>
                                    <p className="text-slate-400">Speaking Rate (Est): {acousticDataMap[msg.message_id].speech_rate_features.estimated_speaking_rate_syllables_per_sec.toFixed(2)} syll/s</p>
                                    <p className="text-slate-400">Pauses (≥150ms): {acousticDataMap[msg.message_id].pause_features.pause_count} ({acousticDataMap[msg.message_id].pause_features.total_pause_duration_seconds.toFixed(2)}s)</p>
                                    <p className="text-slate-400">Pause Ratio: {acousticDataMap[msg.message_id].pause_features.pause_to_speech_ratio !== null && acousticDataMap[msg.message_id].pause_features.pause_to_speech_ratio !== undefined ? acousticDataMap[msg.message_id].pause_features.pause_to_speech_ratio?.toFixed(2) : 'N/A'}</p>
                                  </div>

                                  {/* Spectral Shape */}
                                  <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                    <div className="font-semibold text-amber-400 flex items-center gap-1">
                                      <span>Spectral & Timbre</span>
                                    </div>
                                    <p className="text-slate-300">Centroid: <strong className="font-mono text-white">{acousticDataMap[msg.message_id].spectral_features.spectral_centroid_mean.toFixed(0)} Hz</strong></p>
                                    <p className="text-slate-400">Bandwidth: {acousticDataMap[msg.message_id].spectral_features.spectral_bandwidth_mean.toFixed(0)} Hz</p>
                                    <p className="text-slate-400">Rolloff (85%): {acousticDataMap[msg.message_id].spectral_features.spectral_rolloff_mean.toFixed(0)} Hz</p>
                                    <p className="text-slate-400">ZCR: {acousticDataMap[msg.message_id].spectral_features.zcr_mean.toFixed(4)}</p>
                                  </div>
                                </div>

                                <div className="text-[9px] text-slate-500 italic pt-1 border-t border-slate-800 flex items-center gap-1">
                                  <span>Descriptive acoustic signal measurements only. No diagnostic or psychiatric classification.</span>
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Milestone 7: NLP Analysis on Transcribed Voice */}
                          {msg.transcript_text && (
                            <div className="pt-2 border-t border-slate-700/60 space-y-2 text-xs">
                              <div className="flex items-center justify-between gap-2">
                                <span className="text-[10px] text-slate-400 font-mono flex items-center gap-1">
                                  <BookOpen className="w-3 h-3 text-indigo-400" />
                                  Transcript NLP (M7)
                                </span>
                                <div className="flex items-center gap-1.5">
                                  {analyzingNLPMap[msg.message_id] ? (
                                    <span className="flex items-center gap-1 text-[10px] text-indigo-300 font-mono animate-pulse">
                                      <Loader2 className="w-3 h-3 animate-spin" />
                                      Analyzing...
                                    </span>
                                  ) : nlpDataMap[msg.message_id] || msg.has_nlp_features ? (
                                    <button
                                      onClick={() => toggleNLPDrawer(msg.message_id)}
                                      className="px-2 py-0.5 rounded bg-indigo-950/70 hover:bg-indigo-900/80 text-indigo-300 text-[10px] font-mono border border-indigo-800/60 flex items-center gap-1 cursor-pointer transition-colors"
                                    >
                                      <BookOpen className="w-2.5 h-2.5" />
                                      {expandedNLPMap[msg.message_id] ? 'Hide 66-d' : 'View 66-d'}
                                    </button>
                                  ) : (
                                    <button
                                      onClick={() => handleAnalyzeNLP(msg.message_id)}
                                      className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-indigo-400 hover:text-indigo-300 text-[10px] font-mono border border-slate-700 flex items-center gap-1 cursor-pointer transition-colors"
                                      title="Extract lexical, sentiment, emotion, and topic features from transcript"
                                    >
                                      <Sparkles className="w-2.5 h-2.5" />
                                      Analyze NLP
                                    </button>
                                  )}
                                </div>
                              </div>

                              {/* Collapsible NLP Drawer for Voice Transcript */}
                              {expandedNLPMap[msg.message_id] && nlpDataMap[msg.message_id] && (
                                <div className="bg-slate-900/95 border border-indigo-900/50 rounded-xl p-3 space-y-2.5 text-[11px] shadow-xl text-left">
                                  <div className="flex items-center justify-between text-[9px] text-slate-400 font-mono pb-1 border-b border-slate-800">
                                    <span>Engine: <strong className="text-slate-200">{nlpDataMap[msg.message_id].engine}</strong> ({nlpDataMap[msg.message_id].source_type})</span>
                                    <span className="px-1.5 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 font-bold">
                                      {nlpDataMap[msg.message_id].feature_vector_version} (66 dims)
                                    </span>
                                  </div>

                                  <div className="grid grid-cols-2 gap-2 text-[10px]">
                                    <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                      <div className="font-semibold text-indigo-400 flex items-center gap-1">
                                        <BookOpen className="w-3 h-3" />
                                        <span>Lexical & Diversity</span>
                                      </div>
                                      <p className="text-slate-300">Words / Chars: <strong className="font-mono text-white">{nlpDataMap[msg.message_id].lexical_features.word_count}w / {nlpDataMap[msg.message_id].lexical_features.char_count}c</strong></p>
                                      <p className="text-slate-400">TTR Diversity: <span className="font-mono text-emerald-400">{(nlpDataMap[msg.message_id].lexical_features.type_token_ratio_ttr * 100).toFixed(1)}%</span></p>
                                      <p className="text-slate-400">Avg Word Len: {nlpDataMap[msg.message_id].lexical_features.avg_word_length.toFixed(1)} chars</p>
                                      <p className="text-slate-400">Pronouns: 1st Sing({nlpDataMap[msg.message_id].lexical_features.first_person_singular_count}), 1st Plur({nlpDataMap[msg.message_id].lexical_features.first_person_plural_count}), 2nd({nlpDataMap[msg.message_id].lexical_features.second_person_count}), 3rd({nlpDataMap[msg.message_id].lexical_features.third_person_count})</p>
                                    </div>

                                    <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                      <div className="font-semibold text-emerald-400 flex items-center gap-1">
                                        <Smile className="w-3 h-3" />
                                        <span>Sentiment Polarity</span>
                                      </div>
                                      <p className="text-slate-300">
                                        VADER Compound: <strong className={`font-mono ${nlpDataMap[msg.message_id].sentiment_features.vader_compound > 0.05 ? 'text-emerald-400' : nlpDataMap[msg.message_id].sentiment_features.vader_compound < -0.05 ? 'text-rose-400' : 'text-slate-200'}`}>{nlpDataMap[msg.message_id].sentiment_features.vader_compound.toFixed(2)}</strong>
                                      </p>
                                      <p className="text-slate-400">Pos / Neu / Neg: <span className="font-mono text-white">{(nlpDataMap[msg.message_id].sentiment_features.sentiment_pos_prob * 100).toFixed(0)}% / {(nlpDataMap[msg.message_id].sentiment_features.sentiment_neu_prob * 100).toFixed(0)}% / {(nlpDataMap[msg.message_id].sentiment_features.sentiment_neg_prob * 100).toFixed(0)}%</span></p>
                                      <p className="text-slate-400">Polarity Score: <span className="font-mono">{nlpDataMap[msg.message_id].sentiment_features.sentiment_polarity_score.toFixed(2)}</span></p>
                                    </div>

                                    <div className="col-span-2 bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1.5">
                                      <div className="font-semibold text-amber-400 flex items-center justify-between">
                                        <div className="flex items-center gap-1">
                                          <Sparkles className="w-3 h-3" />
                                          <span>Affective Emotion Distribution</span>
                                        </div>
                                        <span className="text-[9px] text-slate-500 font-mono">Sum = 100%</span>
                                      </div>
                                      <div className="grid grid-cols-7 gap-1 text-center text-[9px] font-mono">
                                        <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                          <span className="text-emerald-400 block font-bold">Joy</span>
                                          <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_joy * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                          <span className="text-blue-400 block font-bold">Sad</span>
                                          <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_sadness * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                          <span className="text-rose-400 block font-bold">Anger</span>
                                          <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_anger * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                          <span className="text-amber-400 block font-bold">Fear</span>
                                          <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_fear * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                          <span className="text-purple-400 block font-bold">Surpr</span>
                                          <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_surprise * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                          <span className="text-orange-400 block font-bold">Disg</span>
                                          <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_disgust * 100).toFixed(0)}%</span>
                                        </div>
                                        <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                          <span className="text-slate-400 block font-bold">Neut</span>
                                          <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_neutral * 100).toFixed(0)}%</span>
                                        </div>
                                      </div>
                                    </div>
                                  </div>

                                  <div className="text-[9px] text-slate-500 italic pt-1 border-t border-slate-800 flex items-center gap-1">
                                    <span>Descriptive linguistic measurements only. No diagnostic or psychiatric classification.</span>
                                  </div>
                                </div>
                              )}
                            </div>
                          )}


                          {/* Milestone 9: Personal Baseline & Temporal Trend Section for Voice */}
                          <div className="pt-2 border-t border-slate-700/60 space-y-2 text-xs">
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-[10px] text-slate-400 font-mono flex items-center gap-1">
                                <TrendingUp className="w-3 h-3 text-cyan-400" />
                                Temporal Deviation (M9)
                              </span>
                              <div className="flex items-center gap-1.5">
                                {analyzingTemporalMap[msg.message_id] ? (
                                  <span className="flex items-center gap-1 text-[10px] text-cyan-300 font-mono animate-pulse">
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                    Evaluating...
                                  </span>
                                ) : temporalDataMap[msg.message_id] || msg.has_temporal_trends ? (
                                  <button
                                    onClick={() => toggleTemporalDrawer(msg.message_id)}
                                    className="px-2 py-0.5 rounded bg-cyan-950/70 hover:bg-cyan-900/80 text-cyan-300 text-[10px] font-mono border border-cyan-800/60 flex items-center gap-1 cursor-pointer transition-colors"
                                  >
                                    <Activity className="w-2.5 h-2.5" />
                                    {expandedTemporalMap[msg.message_id] ? 'Hide Deviation' : 'View Deviation'}
                                  </button>
                                ) : (
                                  <button
                                    onClick={() => handleEvaluateTemporal(msg.message_id)}
                                    className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300 text-[10px] font-mono border border-slate-700 flex items-center gap-1 cursor-pointer transition-colors"
                                    title="Evaluate message z-scores and longitudinal trends against strictly preceding personal baseline"
                                  >
                                    <Compass className="w-2.5 h-2.5" />
                                    Evaluate M9
                                  </button>
                                )}
                              </div>
                            </div>

                            {/* Collapsible Temporal Trend Drawer for Voice */}
                            {expandedTemporalMap[msg.message_id] && temporalDataMap[msg.message_id] && (
                              <div className="bg-slate-900/95 border border-cyan-900/50 rounded-xl p-3 space-y-2.5 text-[11px] shadow-xl text-left">
                                <div className="flex items-center justify-between text-[9px] text-slate-400 font-mono pb-1 border-b border-slate-800">
                                  <span>Status: <strong className={temporalDataMap[msg.message_id].is_baseline_established ? 'text-emerald-400' : 'text-amber-400'}>{temporalDataMap[msg.message_id].status}</strong></span>
                                  <span className={`px-1.5 py-0.5 rounded border font-bold ${temporalDataMap[msg.message_id].is_baseline_established ? 'bg-emerald-950 text-emerald-300 border-emerald-800' : 'bg-amber-950 text-amber-300 border-amber-800'}`}>
                                    {temporalDataMap[msg.message_id].is_baseline_established ? `Baseline Established (N=${temporalDataMap[msg.message_id].sample_count_at_eval})` : `Calibrating (N=${temporalDataMap[msg.message_id].sample_count_at_eval}/3)`}
                                  </span>
                                </div>

                                <div className="flex flex-wrap gap-1.5 text-[9px] font-mono">
                                  <span className="px-2 py-0.5 rounded-full bg-slate-950 text-slate-300 border border-slate-800">
                                    Valid Dims: <strong className="text-cyan-400">{temporalDataMap[msg.message_id].valid_dimension_count} / 140</strong>
                                  </span>
                                  <span className="px-2 py-0.5 rounded-full bg-slate-950 text-slate-400 border border-slate-800">
                                    Mean |Z|: <strong className="text-white">{temporalDataMap[msg.message_id].mean_absolute_z.toFixed(2)}σ</strong>
                                  </span>
                                  <span className="px-2 py-0.5 rounded-full bg-slate-950 text-slate-400 border border-slate-800">
                                    RMS Z: <strong className="text-white">{temporalDataMap[msg.message_id].rms_z.toFixed(2)}σ</strong>
                                  </span>
                                  <span className="px-2 py-0.5 rounded-full bg-slate-950 text-slate-400 border border-slate-800">
                                    Std Euclidean: <strong className="text-white">{temporalDataMap[msg.message_id].standardized_euclidean_distance.toFixed(2)}</strong>
                                  </span>
                                </div>

                                {temporalDataMap[msg.message_id].top_deviations && temporalDataMap[msg.message_id].top_deviations.length > 0 ? (
                                  <div className="space-y-1.5">
                                    <span className="text-[9px] font-mono text-cyan-400 font-semibold block">Top Behavioral Deviations (|Z| Descending):</span>
                                    <div className="space-y-1">
                                      {temporalDataMap[msg.message_id].top_deviations.slice(0, 3).map((item, idx) => (
                                        <div key={idx} className="p-1.5 rounded bg-slate-950/80 border border-slate-800/80 flex items-center justify-between text-[10px] font-mono">
                                          <div>
                                            <span className="text-white font-bold block">{item.name}</span>
                                            <span className="text-[9px] text-slate-400">Val: {item.raw_value.toFixed(2)} | Base: {item.baseline_mean.toFixed(2)} ± {item.baseline_std.toFixed(2)}</span>
                                          </div>
                                          <span className={`px-1.5 py-0.5 rounded font-bold ${Math.abs(item.clipped_zscore) >= 2.0 ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-slate-900 text-slate-300 border border-slate-800'}`}>
                                            {item.clipped_zscore > 0 ? `+${item.clipped_zscore.toFixed(2)}` : item.clipped_zscore.toFixed(2)}σ
                                          </span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                ) : (
                                  <p className="text-[10px] text-slate-500 italic">No significant deviations or baseline still calibrating.</p>
                                )}

                                <div className="text-[9px] text-slate-500 italic pt-1 border-t border-slate-800 flex items-center gap-1">
                                  <Shield className="w-2.5 h-2.5 text-cyan-400 shrink-0" />
                                  <span>Descriptive within-subject statistical variations only. No clinical or diagnostic classification.</span>
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Milestone 8: Multimodal Feature Fusion Section for Voice */}
                          <div className="pt-2 border-t border-slate-700/60 space-y-2 text-xs">
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-[10px] text-slate-400 font-mono flex items-center gap-1">
                                <Layers className="w-3 h-3 text-purple-400" />
                                Multimodal Fusion (M8)
                              </span>
                              <div className="flex items-center gap-1.5">
                                {fusingMultimodalMap[msg.message_id] ? (
                                  <span className="flex items-center gap-1 text-[10px] text-purple-300 font-mono animate-pulse">
                                    <Loader2 className="w-3 h-3 animate-spin" />
                                    Fusing 140-d...
                                  </span>
                                ) : multimodalDataMap[msg.message_id] || msg.has_multimodal_features ? (
                                  <button
                                    onClick={() => toggleMultimodalDrawer(msg.message_id)}
                                    className="px-2 py-0.5 rounded bg-purple-950/70 hover:bg-purple-900/80 text-purple-300 text-[10px] font-mono border border-purple-800/60 flex items-center gap-1 cursor-pointer transition-colors"
                                  >
                                    <GitMerge className="w-2.5 h-2.5" />
                                    {expandedMultimodalMap[msg.message_id] ? 'Hide 140-d' : 'View 140-d'}
                                  </button>
                                ) : (
                                  <button
                                    onClick={() => handleFuseMultimodal(msg.message_id)}
                                    className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-purple-400 hover:text-purple-300 text-[10px] font-mono border border-slate-700 flex items-center gap-1 cursor-pointer transition-colors"
                                    title="Align and concatenate M3 communication, M6 acoustic, and M7 linguistic features into 140-d vector"
                                  >
                                    <Sliders className="w-2.5 h-2.5" />
                                    Fuse Multimodal
                                  </button>
                                )}
                              </div>
                            </div>

                            {/* Collapsible Multimodal Fusion Drawer */}
                            {expandedMultimodalMap[msg.message_id] && multimodalDataMap[msg.message_id] && (
                              <div className="bg-slate-900/95 border border-purple-900/50 rounded-xl p-3 space-y-2.5 text-[11px] shadow-xl text-left">
                                <div className="flex items-center justify-between text-[9px] text-slate-400 font-mono pb-1 border-b border-slate-800">
                                  <span>Strategy: <strong className="text-slate-200">{multimodalDataMap[msg.message_id].fusion_strategy}</strong></span>
                                  <span className="px-1.5 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 font-bold">
                                    {multimodalDataMap[msg.message_id].feature_vector_version} (140 dims)
                                  </span>
                                </div>

                                <div className="flex flex-wrap gap-1.5 text-[9px] font-mono">
                                  <span className={`px-2 py-0.5 rounded-full flex items-center gap-1 border ${multimodalDataMap[msg.message_id].has_communication ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800/80' : 'bg-slate-900 text-slate-500 border-slate-800'}`}>
                                    {multimodalDataMap[msg.message_id].has_communication ? <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" /> : <XCircle className="w-2.5 h-2.5 text-slate-600" />}
                                    M3 Comm: 10 dims
                                  </span>
                                  <span className={`px-2 py-0.5 rounded-full flex items-center gap-1 border ${multimodalDataMap[msg.message_id].has_acoustic ? 'bg-cyan-950/80 text-cyan-300 border-cyan-800/80' : 'bg-slate-900 text-slate-500 border-slate-800'}`}>
                                    {multimodalDataMap[msg.message_id].has_acoustic ? <CheckCircle2 className="w-2.5 h-2.5 text-cyan-400" /> : <XCircle className="w-2.5 h-2.5 text-slate-600" />}
                                    M6 Acoustic: {multimodalDataMap[msg.message_id].has_acoustic ? '59 dims' : '0.0 Imputed'}
                                  </span>
                                  <span className={`px-2 py-0.5 rounded-full flex items-center gap-1 border ${multimodalDataMap[msg.message_id].has_nlp ? 'bg-indigo-950/80 text-indigo-300 border-indigo-800/80' : 'bg-slate-900 text-slate-500 border-slate-800'}`}>
                                    {multimodalDataMap[msg.message_id].has_nlp ? <CheckCircle2 className="w-2.5 h-2.5 text-indigo-400" /> : <XCircle className="w-2.5 h-2.5 text-slate-600" />}
                                    M7 NLP: {multimodalDataMap[msg.message_id].has_nlp ? '66 dims' : 'Neutral Imputed'}
                                  </span>
                                </div>

                                <div className="grid grid-cols-2 gap-2 text-[10px]">
                                  <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                    <div className="font-semibold text-emerald-400 flex items-center gap-1">
                                      <BarChart2 className="w-3 h-3" />
                                      <span>Communication [00..09]</span>
                                    </div>
                                    <p className="text-slate-300">Turn: <strong className="font-mono text-white">#{multimodalDataMap[msg.message_id].communication_features.turn_index}</strong></p>
                                    <p className="text-slate-400">Latency: <span className="font-mono text-white">{multimodalDataMap[msg.message_id].communication_features.response_latency_seconds}s</span></p>
                                    <p className="text-slate-400">Ratio: {(multimodalDataMap[msg.message_id].communication_features.sender_message_ratio * 100).toFixed(1)}%</p>
                                  </div>

                                  <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                    <div className="font-semibold text-purple-400 flex items-center gap-1">
                                      <Layers className="w-3 h-3" />
                                      <span>Modality Mask [135..139]</span>
                                    </div>
                                    <p className="text-slate-300">Type: <strong className="font-mono text-white">{multimodalDataMap[msg.message_id].modality_mask.is_voice_message ? 'Voice Message' : 'Text Message'}</strong></p>
                                    <p className="text-slate-400">Vector Len: <span className="font-mono text-emerald-400">140 / 140 finite floats</span></p>
                                  </div>
                                </div>

                                <div className="bg-slate-950/90 p-2 rounded-lg border border-purple-900/40 space-y-1">
                                  <div className="flex items-center justify-between text-[9px] font-mono text-slate-400">
                                    <span className="text-purple-300 font-semibold">Normalized Vector Summary (140-d)</span>
                                    <span className="text-emerald-400">Scale: [0.0, 1.0]</span>
                                  </div>
                                  <p className="text-[10px] text-slate-300 font-mono">
                                    Mean: <strong className="text-white">{(multimodalDataMap[msg.message_id].normalized_fused_vector.reduce((a, b) => a + b, 0) / 140).toFixed(4)}</strong> | Min: <span className="text-emerald-400">{Math.min(...multimodalDataMap[msg.message_id].normalized_fused_vector).toFixed(4)}</span> | Max: <span className="text-amber-400">{Math.max(...multimodalDataMap[msg.message_id].normalized_fused_vector).toFixed(4)}</span>
                                  </p>
                                </div>

                                <div className="text-[9px] text-slate-500 italic pt-1 border-t border-slate-800 flex items-center gap-1">
                                  <Shield className="w-2.5 h-2.5 text-purple-400 shrink-0" />
                                  <span>Aligned statistical representation for research screening only. No clinical or diagnostic ratings.</span>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ) : (
                        <div className="flex flex-col space-y-1">
                          <div
                            className={`max-w-[80%] sm:max-w-lg rounded-2xl px-4 py-2.5 text-xs sm:text-sm leading-relaxed shadow-lg ${
                              isSelf
                                ? 'bg-emerald-600 text-white rounded-br-none border border-emerald-500/40'
                                : 'bg-slate-800/90 text-slate-100 rounded-bl-none border border-slate-700'
                            }`}
                          >
                            <p className="whitespace-pre-wrap break-words">{msg.message_text}</p>
                          </div>

                          {/* Milestone 7: NLP Analysis on Text Messages */}
                          <div className={`flex items-center gap-1.5 text-[10px] font-mono mt-0.5 ${isSelf ? 'justify-end' : 'justify-start'}`}>
                            {analyzingNLPMap[msg.message_id] ? (
                              <span className="flex items-center gap-1 text-indigo-300 animate-pulse">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                Analyzing NLP...
                              </span>
                            ) : nlpDataMap[msg.message_id] || msg.has_nlp_features ? (
                              <button
                                onClick={() => toggleNLPDrawer(msg.message_id)}
                                className="px-2 py-0.5 rounded bg-indigo-950/70 hover:bg-indigo-900/80 text-indigo-300 border border-indigo-800/60 flex items-center gap-1 cursor-pointer transition-colors"
                              >
                                <BookOpen className="w-2.5 h-2.5" />
                                {expandedNLPMap[msg.message_id] ? 'Hide NLP (66-d)' : 'View NLP (66-d)'}
                              </button>
                            ) : (
                              <button
                                onClick={() => handleAnalyzeNLP(msg.message_id)}
                                className="px-2 py-0.5 rounded bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-indigo-300 border border-slate-800 flex items-center gap-1 cursor-pointer transition-colors"
                                title="Extract lexical diversity, sentiment, emotion, and topic features"
                              >
                                <Sparkles className="w-2.5 h-2.5 text-indigo-400" />
                                <span>Analyze NLP</span>
                              </button>
                            )}
                          </div>

                          {/* Collapsible NLP Drawer for Text Message */}
                          {expandedNLPMap[msg.message_id] && nlpDataMap[msg.message_id] && (
                            <div className="max-w-[85%] sm:max-w-md bg-slate-900/95 border border-indigo-900/50 rounded-xl p-3 space-y-2.5 text-[11px] shadow-xl text-left mt-1">
                              <div className="flex items-center justify-between text-[9px] text-slate-400 font-mono pb-1 border-b border-slate-800">
                                <span>Engine: <strong className="text-slate-200">{nlpDataMap[msg.message_id].engine}</strong></span>
                                <span className="px-1.5 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 font-bold">
                                  {nlpDataMap[msg.message_id].feature_vector_version} (66 dims)
                                </span>
                              </div>

                              <div className="grid grid-cols-2 gap-2 text-[10px]">
                                <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                  <div className="font-semibold text-indigo-400 flex items-center gap-1">
                                    <BookOpen className="w-3 h-3" />
                                    <span>Lexical & Diversity</span>
                                  </div>
                                  <p className="text-slate-300">Words / Chars: <strong className="font-mono text-white">{nlpDataMap[msg.message_id].lexical_features.word_count}w / {nlpDataMap[msg.message_id].lexical_features.char_count}c</strong></p>
                                  <p className="text-slate-400">TTR Diversity: <span className="font-mono text-emerald-400">{(nlpDataMap[msg.message_id].lexical_features.type_token_ratio_ttr * 100).toFixed(1)}%</span></p>
                                  <p className="text-slate-400">Avg Word Len: {nlpDataMap[msg.message_id].lexical_features.avg_word_length.toFixed(1)} chars</p>
                                  <p className="text-slate-400">Pronouns: 1st Sing({nlpDataMap[msg.message_id].lexical_features.first_person_singular_count}), 1st Plur({nlpDataMap[msg.message_id].lexical_features.first_person_plural_count}), 2nd({nlpDataMap[msg.message_id].lexical_features.second_person_count}), 3rd({nlpDataMap[msg.message_id].lexical_features.third_person_count})</p>
                                </div>

                                <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                  <div className="font-semibold text-emerald-400 flex items-center gap-1">
                                    <Smile className="w-3 h-3" />
                                    <span>Sentiment Polarity</span>
                                  </div>
                                  <p className="text-slate-300">
                                    VADER Compound: <strong className={`font-mono ${nlpDataMap[msg.message_id].sentiment_features.vader_compound > 0.05 ? 'text-emerald-400' : nlpDataMap[msg.message_id].sentiment_features.vader_compound < -0.05 ? 'text-rose-400' : 'text-slate-200'}`}>{nlpDataMap[msg.message_id].sentiment_features.vader_compound.toFixed(2)}</strong>
                                  </p>
                                  <p className="text-slate-400">Pos / Neu / Neg: <span className="font-mono text-white">{(nlpDataMap[msg.message_id].sentiment_features.sentiment_pos_prob * 100).toFixed(0)}% / {(nlpDataMap[msg.message_id].sentiment_features.sentiment_neu_prob * 100).toFixed(0)}% / {(nlpDataMap[msg.message_id].sentiment_features.sentiment_neg_prob * 100).toFixed(0)}%</span></p>
                                  <p className="text-slate-400">Polarity Score: <span className="font-mono">{nlpDataMap[msg.message_id].sentiment_features.sentiment_polarity_score.toFixed(2)}</span></p>
                                </div>

                                <div className="col-span-2 bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1.5">
                                  <div className="font-semibold text-amber-400 flex items-center justify-between">
                                    <div className="flex items-center gap-1">
                                      <Sparkles className="w-3 h-3" />
                                      <span>Affective Emotion Distribution</span>
                                    </div>
                                    <span className="text-[9px] text-slate-500 font-mono">Sum = 100%</span>
                                  </div>
                                  <div className="grid grid-cols-7 gap-1 text-center text-[9px] font-mono">
                                    <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                      <span className="text-emerald-400 block font-bold">Joy</span>
                                      <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_joy * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                      <span className="text-blue-400 block font-bold">Sad</span>
                                      <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_sadness * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                      <span className="text-rose-400 block font-bold">Anger</span>
                                      <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_anger * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                      <span className="text-amber-400 block font-bold">Fear</span>
                                      <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_fear * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                      <span className="text-purple-400 block font-bold">Surpr</span>
                                      <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_surprise * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                      <span className="text-orange-400 block font-bold">Disg</span>
                                      <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_disgust * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="bg-slate-900 p-1 rounded border border-slate-800">
                                      <span className="text-slate-400 block font-bold">Neut</span>
                                      <span className="text-white">{(nlpDataMap[msg.message_id].emotion_features.emotion_neutral * 100).toFixed(0)}%</span>
                                    </div>
                                  </div>
                                </div>
                              </div>

                              <div className="text-[9px] text-slate-500 italic pt-1 border-t border-slate-800 flex items-center gap-1">
                                <span>Descriptive linguistic measurements only. No diagnostic or psychiatric classification.</span>
                              </div>
                            </div>
                          )}


                          {/* Milestone 9: Personal Baseline & Temporal Trend Section for Text */}
                          <div className={`flex items-center gap-1.5 text-[10px] font-mono mt-1 ${isSelf ? 'justify-end' : 'justify-start'}`}>
                            {analyzingTemporalMap[msg.message_id] ? (
                              <span className="flex items-center gap-1 text-cyan-300 animate-pulse">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                Evaluating M9...
                              </span>
                            ) : temporalDataMap[msg.message_id] || msg.has_temporal_trends ? (
                              <button
                                onClick={() => toggleTemporalDrawer(msg.message_id)}
                                className="px-2 py-0.5 rounded bg-cyan-950/70 hover:bg-cyan-900/80 text-cyan-300 border border-cyan-800/60 flex items-center gap-1 cursor-pointer transition-colors"
                              >
                                <Activity className="w-2.5 h-2.5" />
                                {expandedTemporalMap[msg.message_id] ? 'Hide Deviation' : 'View Deviation'}
                              </button>
                            ) : (
                              <button
                                onClick={() => handleEvaluateTemporal(msg.message_id)}
                                className="px-2 py-0.5 rounded bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-cyan-300 border border-slate-800 flex items-center gap-1 cursor-pointer transition-colors"
                                title="Evaluate message z-scores and longitudinal trends against strictly preceding personal baseline"
                              >
                                <Compass className="w-2.5 h-2.5 text-cyan-400" />
                                <span>Evaluate M9</span>
                              </button>
                            )}
                          </div>

                          {/* Collapsible Temporal Trend Drawer for Text */}
                          {expandedTemporalMap[msg.message_id] && temporalDataMap[msg.message_id] && (
                            <div className="max-w-[85%] sm:max-w-md bg-slate-900/95 border border-cyan-900/50 rounded-xl p-3 space-y-2.5 text-[11px] shadow-xl text-left mt-1">
                              <div className="flex items-center justify-between text-[9px] text-slate-400 font-mono pb-1 border-b border-slate-800">
                                <span>Status: <strong className={temporalDataMap[msg.message_id].is_baseline_established ? 'text-emerald-400' : 'text-amber-400'}>{temporalDataMap[msg.message_id].status}</strong></span>
                                <span className={`px-1.5 py-0.5 rounded border font-bold ${temporalDataMap[msg.message_id].is_baseline_established ? 'bg-emerald-950 text-emerald-300 border-emerald-800' : 'bg-amber-950 text-amber-300 border-amber-800'}`}>
                                  {temporalDataMap[msg.message_id].is_baseline_established ? `Baseline Established (N=${temporalDataMap[msg.message_id].sample_count_at_eval})` : `Calibrating (N=${temporalDataMap[msg.message_id].sample_count_at_eval}/3)`}
                                </span>
                              </div>

                              <div className="flex flex-wrap gap-1.5 text-[9px] font-mono">
                                <span className="px-2 py-0.5 rounded-full bg-slate-950 text-slate-300 border border-slate-800">
                                  Valid Dims: <strong className="text-cyan-400">{temporalDataMap[msg.message_id].valid_dimension_count} / 140</strong>
                                </span>
                                <span className="px-2 py-0.5 rounded-full bg-slate-950 text-slate-400 border border-slate-800">
                                  Mean |Z|: <strong className="text-white">{temporalDataMap[msg.message_id].mean_absolute_z.toFixed(2)}σ</strong>
                                </span>
                                <span className="px-2 py-0.5 rounded-full bg-slate-950 text-slate-400 border border-slate-800">
                                  RMS Z: <strong className="text-white">{temporalDataMap[msg.message_id].rms_z.toFixed(2)}σ</strong>
                                </span>
                                <span className="px-2 py-0.5 rounded-full bg-slate-950 text-slate-400 border border-slate-800">
                                  Std Euclidean: <strong className="text-white">{temporalDataMap[msg.message_id].standardized_euclidean_distance.toFixed(2)}</strong>
                                </span>
                              </div>

                              {temporalDataMap[msg.message_id].top_deviations && temporalDataMap[msg.message_id].top_deviations.length > 0 ? (
                                <div className="space-y-1.5">
                                  <span className="text-[9px] font-mono text-cyan-400 font-semibold block">Top Behavioral Deviations (|Z| Descending):</span>
                                  <div className="space-y-1">
                                    {temporalDataMap[msg.message_id].top_deviations.slice(0, 3).map((item, idx) => (
                                      <div key={idx} className="p-1.5 rounded bg-slate-950/80 border border-slate-800/80 flex items-center justify-between text-[10px] font-mono">
                                        <div>
                                          <span className="text-white font-bold block">{item.name}</span>
                                          <span className="text-[9px] text-slate-400">Val: {item.raw_value.toFixed(2)} | Base: {item.baseline_mean.toFixed(2)} ± {item.baseline_std.toFixed(2)}</span>
                                        </div>
                                        <span className={`px-1.5 py-0.5 rounded font-bold ${Math.abs(item.clipped_zscore) >= 2.0 ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-slate-900 text-slate-300 border border-slate-800'}`}>
                                          {item.clipped_zscore > 0 ? `+${item.clipped_zscore.toFixed(2)}` : item.clipped_zscore.toFixed(2)}σ
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              ) : (
                                <p className="text-[10px] text-slate-500 italic">No significant deviations or baseline still calibrating.</p>
                              )}

                              <div className="text-[9px] text-slate-500 italic pt-1 border-t border-slate-800 flex items-center gap-1">
                                <Shield className="w-2.5 h-2.5 text-cyan-400 shrink-0" />
                                <span>Descriptive within-subject statistical variations only. No clinical or diagnostic classification.</span>
                              </div>
                            </div>
                          )}

                          {/* Milestone 8: Multimodal Feature Fusion Section for Text */}
                          <div className={`flex items-center gap-1.5 text-[10px] font-mono mt-1 ${isSelf ? 'justify-end' : 'justify-start'}`}>
                            {fusingMultimodalMap[msg.message_id] ? (
                              <span className="flex items-center gap-1 text-purple-300 animate-pulse">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                Fusing 140-d...
                              </span>
                            ) : multimodalDataMap[msg.message_id] || msg.has_multimodal_features ? (
                              <button
                                onClick={() => toggleMultimodalDrawer(msg.message_id)}
                                className="px-2 py-0.5 rounded bg-purple-950/70 hover:bg-purple-900/80 text-purple-300 border border-purple-800/60 flex items-center gap-1 cursor-pointer transition-colors"
                              >
                                <GitMerge className="w-2.5 h-2.5" />
                                {expandedMultimodalMap[msg.message_id] ? 'Hide 140-d' : 'View 140-d'}
                              </button>
                            ) : (
                              <button
                                onClick={() => handleFuseMultimodal(msg.message_id)}
                                className="px-2 py-0.5 rounded bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-purple-300 border border-slate-800 flex items-center gap-1 cursor-pointer transition-colors"
                                title="Align and concatenate M3 communication, M6 acoustic (0-imputed), and M7 NLP into 140-d vector"
                              >
                                <Sliders className="w-2.5 h-2.5 text-purple-400" />
                                <span>Fuse Multimodal</span>
                              </button>
                            )}
                          </div>

                          {/* Collapsible Multimodal Fusion Drawer for Text Message */}
                          {expandedMultimodalMap[msg.message_id] && multimodalDataMap[msg.message_id] && (
                            <div className="max-w-[85%] sm:max-w-md bg-slate-900/95 border border-purple-900/50 rounded-xl p-3 space-y-2.5 text-[11px] shadow-xl text-left mt-1">
                              <div className="flex items-center justify-between text-[9px] text-slate-400 font-mono pb-1 border-b border-slate-800">
                                <span>Strategy: <strong className="text-slate-200">{multimodalDataMap[msg.message_id].fusion_strategy}</strong></span>
                                <span className="px-1.5 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 font-bold">
                                  {multimodalDataMap[msg.message_id].feature_vector_version} (140 dims)
                                </span>
                              </div>

                              <div className="flex flex-wrap gap-1.5 text-[9px] font-mono">
                                <span className={`px-2 py-0.5 rounded-full flex items-center gap-1 border ${multimodalDataMap[msg.message_id].has_communication ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800/80' : 'bg-slate-900 text-slate-500 border-slate-800'}`}>
                                  {multimodalDataMap[msg.message_id].has_communication ? <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" /> : <XCircle className="w-2.5 h-2.5 text-slate-600" />}
                                  M3 Comm: 10 dims
                                </span>
                                <span className={`px-2 py-0.5 rounded-full flex items-center gap-1 border ${multimodalDataMap[msg.message_id].has_acoustic ? 'bg-cyan-950/80 text-cyan-300 border-cyan-800/80' : 'bg-slate-900 text-slate-500 border-slate-800'}`}>
                                  {multimodalDataMap[msg.message_id].has_acoustic ? <CheckCircle2 className="w-2.5 h-2.5 text-cyan-400" /> : <XCircle className="w-2.5 h-2.5 text-slate-600" />}
                                  M6 Acoustic: 0.0 Imputed
                                </span>
                                <span className={`px-2 py-0.5 rounded-full flex items-center gap-1 border ${multimodalDataMap[msg.message_id].has_nlp ? 'bg-indigo-950/80 text-indigo-300 border-indigo-800/80' : 'bg-slate-900 text-slate-500 border-slate-800'}`}>
                                  {multimodalDataMap[msg.message_id].has_nlp ? <CheckCircle2 className="w-2.5 h-2.5 text-indigo-400" /> : <XCircle className="w-2.5 h-2.5 text-slate-600" />}
                                  M7 NLP: {multimodalDataMap[msg.message_id].has_nlp ? '66 dims' : 'Neutral'}
                                </span>
                              </div>

                              <div className="grid grid-cols-2 gap-2 text-[10px]">
                                <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                  <div className="font-semibold text-emerald-400 flex items-center gap-1">
                                    <BarChart2 className="w-3 h-3" />
                                    <span>Communication [00..09]</span>
                                  </div>
                                  <p className="text-slate-300">Turn: <strong className="font-mono text-white">#{multimodalDataMap[msg.message_id].communication_features.turn_index}</strong></p>
                                  <p className="text-slate-400">Latency: <span className="font-mono text-white">{multimodalDataMap[msg.message_id].communication_features.response_latency_seconds}s</span></p>
                                  <p className="text-slate-400">Ratio: {(multimodalDataMap[msg.message_id].communication_features.sender_message_ratio * 100).toFixed(1)}%</p>
                                </div>

                                <div className="bg-slate-950/70 p-2 rounded-lg border border-slate-800/80 space-y-1">
                                  <div className="font-semibold text-purple-400 flex items-center gap-1">
                                    <Layers className="w-3 h-3" />
                                    <span>Modality Mask [135..139]</span>
                                  </div>
                                  <p className="text-slate-300">Type: <strong className="font-mono text-white">Text Message</strong></p>
                                  <p className="text-slate-400">Vector Len: <span className="font-mono text-emerald-400">140 / 140 finite floats</span></p>
                                </div>
                              </div>

                              <div className="bg-slate-950/90 p-2 rounded-lg border border-purple-900/40 space-y-1">
                                <div className="flex items-center justify-between text-[9px] font-mono text-slate-400">
                                  <span className="text-purple-300 font-semibold">Normalized Vector Summary (140-d)</span>
                                  <span className="text-emerald-400">Scale: [0.0, 1.0]</span>
                                </div>
                                <p className="text-[10px] text-slate-300 font-mono">
                                  Mean: <strong className="text-white">{(multimodalDataMap[msg.message_id].normalized_fused_vector.reduce((a, b) => a + b, 0) / 140).toFixed(4)}</strong> | Min: <span className="text-emerald-400">{Math.min(...multimodalDataMap[msg.message_id].normalized_fused_vector).toFixed(4)}</span> | Max: <span className="text-amber-400">{Math.max(...multimodalDataMap[msg.message_id].normalized_fused_vector).toFixed(4)}</span>
                                </p>
                              </div>

                              <div className="text-[9px] text-slate-500 italic pt-1 border-t border-slate-800 flex items-center gap-1">
                                <Shield className="w-2.5 h-2.5 text-purple-400 shrink-0" />
                                <span>Aligned statistical representation for research screening only. No clinical or diagnostic ratings.</span>
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      <span className="text-[9px] font-mono text-slate-600 mt-0.5 px-1">
                        ID: {msg.message_id.slice(0, 8)}... {msg.message_type === 'voice' ? '• [Voice]' : ''}
                      </span>
                    </div>
                  );
                })
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Footer */}
            <div className="p-3 sm:p-4 bg-slate-950/80 border-t border-slate-800 space-y-2">
              <div className="flex items-end gap-2">
                {/* Voice Recorder Control */}
                <VoiceRecorder
                  roomId={roomId}
                  userId={userId}
                  disabled={connectionStatus !== 'connected'}
                />

                <div className="relative flex-1">
                  <textarea
                    ref={textareaRef}
                    rows={2}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type a message or record voice... (Press Enter to send)"
                    disabled={connectionStatus !== 'connected'}
                    className="w-full resize-none px-4 py-2.5 rounded-xl bg-slate-900 border border-slate-800 text-xs sm:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 disabled:opacity-50 font-sans"
                  />
                  <div className="absolute right-3 bottom-2 text-[10px] font-mono text-slate-500">
                    {inputText.length} / 4000
                  </div>
                </div>

                <button
                  onClick={handleSendMessage}
                  disabled={connectionStatus !== 'connected' || !inputText.trim()}
                  className="p-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:hover:bg-emerald-600 text-white shadow-lg shadow-emerald-950/50 transition-all cursor-pointer"
                  title="Send message (Enter)"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>

              <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1 px-1">
                <span>Persistence: SQLite / PostgreSQL ORM & Audio Store active</span>
                <span className="flex items-center gap-1">
                  <CornerDownLeft className="w-2.5 h-2.5" />
                  <span>Enter to send</span>
                </span>
              </div>
            </div>
          </div>

          {/* Collapsible Communication Metrics Panel */}
          {showAnalytics && (
            <div className="pt-2">
              <CommunicationAnalysisView initialRoomId={roomId} />
            </div>
          )}
        </div>
      )}
      {/* Milestone 10 Analytical Report Modal */}
      {isReportModalOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-4xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="px-6 py-4 bg-slate-950 border-b border-slate-800 flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
                  <FileText className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-white text-base sm:text-lg flex items-center gap-2">
                    <span>Longitudinal Analytical Report</span>
                    <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-indigo-300 font-mono border border-slate-700">
                      v1.0-140d
                    </span>
                  </h3>
                  <p className="text-xs text-slate-400">
                    Participant ID: <span className="font-mono text-slate-200">{userId}</span> | Room: <span className="font-mono text-slate-200">{roomId}</span>
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsReportModalOpen(false)}
                className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-all cursor-pointer"
              >
                <XCircle className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto space-y-6 flex-1 text-sm text-slate-200">
              {isLoadingReport ? (
                <div className="py-16 text-center space-y-3">
                  <Loader2 className="w-8 h-8 text-indigo-400 animate-spin mx-auto" />
                  <p className="text-sm font-semibold text-slate-300">Assembling longitudinal statistical report...</p>
                  <p className="text-xs text-slate-500">Aggregating M9 personal baselines, validity masks, and temporal trajectories.</p>
                </div>
              ) : reportError ? (
                <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 space-y-2">
                  <div className="flex items-center gap-2 font-semibold text-rose-200">
                    <AlertCircle className="w-4 h-4" />
                    <span>Report Generation Notice</span>
                  </div>
                  <p className="text-xs">{reportError}</p>
                </div>
              ) : currentReport ? (
                <>
                  {/* Executive Summary & Classification */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                      <span className="text-[11px] text-slate-400 font-medium">Observations Analyzed</span>
                      <div className="text-2xl font-bold text-white font-mono">{currentReport.metadata.observation_count} <span className="text-xs font-normal text-slate-400">turns</span></div>
                      <span className="text-[11px] text-slate-500">Scope: {currentReport.metadata.report_type}</span>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                      <span className="text-[11px] text-slate-400 font-medium">Personal Baseline Status</span>
                      <div className="flex items-center gap-1.5 text-base font-bold">
                        <span className={`w-2.5 h-2.5 rounded-full ${currentReport.baseline.is_baseline_established ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                        <span className={currentReport.baseline.is_baseline_established ? 'text-emerald-300' : 'text-amber-300'}>
                          {currentReport.baseline.status.toUpperCase()}
                        </span>
                      </div>
                      <span className="text-[11px] text-slate-500 font-mono">N={currentReport.baseline.sample_count} prior observations</span>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-1">
                      <span className="text-[11px] text-slate-400 font-medium">Statistical Variation Category</span>
                      <div className="text-sm font-bold text-indigo-300">
                        {currentReport.deviation_summary.statistical_variation_category}
                      </div>
                      <span className="text-[11px] text-slate-500 line-clamp-1">{currentReport.deviation_summary.category_description}</span>
                    </div>
                  </div>

                  {/* Modality Completeness */}
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                      <Layers className="w-4 h-4 text-emerald-400" />
                      <span>Modality Completeness & Coverage</span>
                    </h4>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800/80 space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300 font-semibold">Communication</span>
                          <span className="font-mono text-emerald-400">{currentReport.modality_coverage.communication.percentage}%</span>
                        </div>
                        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${currentReport.modality_coverage.communication.percentage}%` }} />
                        </div>
                        <span className="text-[10px] text-slate-500">{currentReport.modality_coverage.communication.measured_count} / {currentReport.modality_coverage.communication.total_count} turns measured</span>
                      </div>

                      <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800/80 space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300 font-semibold">Acoustic Signal</span>
                          <span className="font-mono text-cyan-400">{currentReport.modality_coverage.acoustic.percentage}%</span>
                        </div>
                        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${currentReport.modality_coverage.acoustic.percentage}%` }} />
                        </div>
                        <span className="text-[10px] text-slate-500">{currentReport.modality_coverage.acoustic.measured_count} / {currentReport.modality_coverage.acoustic.total_count} turns measured</span>
                      </div>

                      <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800/80 space-y-1.5">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-300 font-semibold">NLP & Linguistics</span>
                          <span className="font-mono text-indigo-400">{currentReport.modality_coverage.nlp.percentage}%</span>
                        </div>
                        <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-indigo-500 h-full rounded-full" style={{ width: `${currentReport.modality_coverage.nlp.percentage}%` }} />
                        </div>
                        <span className="text-[10px] text-slate-500">{currentReport.modality_coverage.nlp.measured_count} / {currentReport.modality_coverage.nlp.total_count} turns measured</span>
                      </div>
                    </div>
                  </div>

                  {/* Aggregate Statistical Variation Metrics */}
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-indigo-400" />
                      <span>Aggregate Statistical Variation Metrics</span>
                    </h4>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
                        <span className="text-xs text-slate-400">Mean Absolute |Z|</span>
                        <div className="text-lg font-bold font-mono text-emerald-400">{currentReport.deviation_summary.mean_absolute_z.mean} <span className="text-xs font-normal">σ</span></div>
                        <div className="text-[11px] text-slate-500">Median: {currentReport.deviation_summary.mean_absolute_z.median}σ · Range: [{currentReport.deviation_summary.mean_absolute_z.minimum}..{currentReport.deviation_summary.mean_absolute_z.maximum}]</div>
                      </div>

                      <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
                        <span className="text-xs text-slate-400">RMS Z (Root Mean Sq)</span>
                        <div className="text-lg font-bold font-mono text-cyan-400">{currentReport.deviation_summary.rms_z.mean} <span className="text-xs font-normal">σ</span></div>
                        <div className="text-[11px] text-slate-500">Median: {currentReport.deviation_summary.rms_z.median}σ · Range: [{currentReport.deviation_summary.rms_z.minimum}..{currentReport.deviation_summary.rms_z.maximum}]</div>
                      </div>

                      <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1">
                        <span className="text-xs text-slate-400">Std Euclidean Dist</span>
                        <div className="text-lg font-bold font-mono text-purple-400">{currentReport.deviation_summary.standardized_euclidean_distance.mean}</div>
                        <div className="text-[11px] text-slate-500">Median: {currentReport.deviation_summary.standardized_euclidean_distance.median} · Range: [{currentReport.deviation_summary.standardized_euclidean_distance.minimum}..{currentReport.deviation_summary.standardized_euclidean_distance.maximum}]</div>
                      </div>
                    </div>
                  </div>

                  {/* Top Behavioral Feature Shifts */}
                  {currentReport.feature_shifts.length > 0 && (
                    <div className="space-y-3">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-amber-400" />
                        <span>Largest Behavioral Feature Shifts (|Z| Descending)</span>
                      </h4>
                      <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-950/40">
                        <table className="w-full text-left text-xs border-collapse">
                          <thead>
                            <tr className="border-b border-slate-800 bg-slate-950/80 text-slate-400">
                              <th className="py-2.5 px-3">Idx</th>
                              <th className="py-2.5 px-3">Feature Name</th>
                              <th className="py-2.5 px-3">Modality</th>
                              <th className="py-2.5 px-3">Mean |Z|</th>
                              <th className="py-2.5 px-3">Max |Z|</th>
                              <th className="py-2.5 px-3">Latest Signed Z</th>
                              <th className="py-2.5 px-3">Baseline Ref (μ ± σ)</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/60 font-mono">
                            {currentReport.feature_shifts.map((s) => (
                              <tr key={s.feature_index} className="hover:bg-slate-900/50">
                                <td className="py-2 px-3 text-slate-500">#{s.feature_index.toString().padStart(3, '0')}</td>
                                <td className="py-2 px-3 text-slate-200 font-sans font-medium">{s.feature_name}</td>
                                <td className="py-2 px-3">
                                  <span className={`px-2 py-0.5 rounded text-[10px] uppercase ${
                                    s.modality === 'acoustic' ? 'bg-cyan-950 text-cyan-300 border border-cyan-800' :
                                    s.modality === 'nlp' ? 'bg-indigo-950 text-indigo-300 border border-indigo-800' :
                                    'bg-emerald-950 text-emerald-300 border border-emerald-800'
                                  }`}>
                                    {s.modality}
                                  </span>
                                </td>
                                <td className="py-2 px-3 text-amber-300 font-bold">{s.mean_absolute_z}σ</td>
                                <td className="py-2 px-3 text-slate-300">{s.max_absolute_z}σ</td>
                                <td className="py-2 px-3 text-slate-200">{s.latest_signed_z > 0 ? `+${s.latest_signed_z}` : s.latest_signed_z}σ</td>
                                <td className="py-2 px-3 text-slate-400">{s.baseline_mean} ± {s.baseline_std}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Longitudinal Temporal Trends */}
                  {currentReport.trend_summary.top_trends.length > 0 && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                          <Compass className="w-4 h-4 text-emerald-400" />
                          <span>Longitudinal Temporal Trends (OLS & Moving Averages)</span>
                        </h4>
                        <span className="text-xs text-slate-500">
                          {currentReport.trend_summary.increasing_count} Increasing · {currentReport.trend_summary.decreasing_count} Decreasing · {currentReport.trend_summary.stable_count} Stable
                        </span>
                      </div>
                      <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-950/40">
                        <table className="w-full text-left text-xs border-collapse">
                          <thead>
                            <tr className="border-b border-slate-800 bg-slate-950/80 text-slate-400">
                              <th className="py-2.5 px-3">Feature Name</th>
                              <th className="py-2.5 px-3">Modality</th>
                              <th className="py-2.5 px-3">Trend Direction</th>
                              <th className="py-2.5 px-3">Norm Slope (/hr)</th>
                              <th className="py-2.5 px-3">Latest SMA-3</th>
                              <th className="py-2.5 px-3">Latest EMA-3</th>
                              <th className="py-2.5 px-3">Rate of Change</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/60 font-mono">
                            {currentReport.trend_summary.top_trends.map((t) => (
                              <tr key={t.feature_index} className="hover:bg-slate-900/50">
                                <td className="py-2 px-3 text-slate-200 font-sans font-medium">{t.feature_name}</td>
                                <td className="py-2 px-3">
                                  <span className="text-[10px] text-slate-400 uppercase">{t.modality}</span>
                                </td>
                                <td className="py-2 px-3">
                                  <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold ${
                                    t.direction === 'increasing' ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' :
                                    t.direction === 'decreasing' ? 'bg-rose-950 text-rose-300 border border-rose-800' :
                                    'bg-slate-800 text-slate-300 border border-slate-700'
                                  }`}>
                                    {t.direction === 'increasing' ? '↑ Increasing' : t.direction === 'decreasing' ? '↓ Decreasing' : '→ Stable'}
                                  </span>
                                </td>
                                <td className="py-2 px-3 text-slate-300">{t.normalized_slope > 0 ? `+${t.normalized_slope}` : t.normalized_slope}</td>
                                <td className="py-2 px-3 text-slate-400">{t.latest_sma3 ?? 'N/A'}</td>
                                <td className="py-2 px-3 text-slate-400">{t.latest_ema3 ?? 'N/A'}</td>
                                <td className="py-2 px-3 text-slate-400">{t.latest_roc ?? 'N/A'}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Research Limitations */}
                  <div className="space-y-2 p-4 rounded-xl bg-slate-950/70 border border-slate-800 text-xs">
                    <h5 className="font-bold text-slate-300 flex items-center gap-1.5">
                      <Info className="w-4 h-4 text-indigo-400" />
                      <span>Research Limitations & Statistical Notes</span>
                    </h5>
                    <ul className="list-disc list-inside space-y-1 text-slate-400">
                      {currentReport.limitations.map((limit, idx) => (
                        <li key={idx}>{limit}</li>
                      ))}
                    </ul>
                  </div>

                  {/* Mandatory Research Disclaimer */}
                  <div className="p-4 rounded-xl bg-indigo-950/30 border border-indigo-800/60 text-indigo-200 text-xs space-y-1">
                    <div className="flex items-center gap-1.5 font-bold text-indigo-300">
                      <Shield className="w-4 h-4" />
                      <span>MANDATORY RESEARCH DISCLAIMER</span>
                    </div>
                    <p className="leading-relaxed">{currentReport.disclaimer}</p>
                  </div>
                </>
              ) : null}
            </div>

            {/* Modal Footer / Export Toolbar */}
            {currentReport && (
              <div className="px-6 py-3 bg-slate-950 border-t border-slate-800 flex items-center justify-between gap-4">
                <div className="text-xs text-slate-500 font-mono">
                  Report ID: {currentReport.metadata.report_id.slice(0, 16)}...
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleExport('json')}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-all cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>JSON</span>
                  </button>
                  <button
                    onClick={() => handleExport('markdown')}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-all cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Markdown</span>
                  </button>
                  <button
                    onClick={() => handleExport('html')}
                    className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all cursor-pointer shadow-sm shadow-indigo-950/50"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>HTML</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
};
