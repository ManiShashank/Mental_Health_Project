import React from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  MessageSquare,
  Mic,
  BarChart3,
  Sparkles,
  Shield,
  Layers,
  Clock,
  Binary
} from 'lucide-react';

export const Home: React.FC = () => {
  return (
    <div className="space-y-12 pb-12">
      {/* Hero Section */}
      <section className="relative pt-6 pb-4 sm:pt-10">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-emerald-950/30 via-slate-950 to-slate-950"></div>
        
        <div className="max-w-4xl mx-auto text-center space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Academic Research Project • Milestone 1 Foundation</span>
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Multimodal Analysis of Communication & Language Patterns
          </h1>

          <p className="text-base sm:text-lg text-slate-300 max-w-3xl mx-auto leading-relaxed">
            A personalized pre-screening research platform capturing multi-turn dyadic interactions to measure statistical deviations in language, acoustics, and communicative behavior relative to an individual’s own historical baseline.
          </p>

          {/* Action CTAs */}
          <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
            <Link
              to="/status"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-emerald-600 text-white hover:bg-emerald-500 shadow-lg shadow-emerald-900/30 transition-all hover:scale-105"
            >
              <Activity className="w-4 h-4" />
              <span>Verify System Status</span>
            </Link>

            <Link
              to="/docs"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-slate-800 text-slate-200 hover:bg-slate-700 border border-slate-700 transition-all"
            >
              <Layers className="w-4 h-4" />
              <span>View Specifications & Algorithms</span>
            </Link>
          </div>
        </div>
      </section>

      {/* Non-Diagnostic Clinical Directive Banner */}
      <section className="max-w-5xl mx-auto">
        <div className="rounded-2xl bg-gradient-to-r from-amber-950/40 via-amber-900/20 to-amber-950/40 border border-amber-800/40 p-5 sm:p-6 shadow-xl">
          <div className="flex items-start gap-4">
            <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 flex-shrink-0">
              <Shield className="w-6 h-6" />
            </div>
            <div className="space-y-1.5 text-xs sm:text-sm text-slate-300">
              <h3 className="font-bold text-amber-300 text-sm sm:text-base flex items-center gap-2">
                <span>Analytical Research Boundary & Clinical Notice</span>
              </h3>
              <p className="leading-relaxed">
                This system is strictly an <strong>analytical pre-screening research platform</strong>. It does <strong>NOT</strong> diagnose medical or psychological conditions, provide clinical evaluations, or make unsupported therapeutic claims. The outputs summarize observed communicative, linguistic, and acoustic features and compute statistical z-score deviations against an individual's historical baseline.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 3 Modalities Grid */}
      <section className="max-w-6xl mx-auto space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold tracking-tight text-white">Tri-Modal Analytical Framework</h2>
          <p className="text-sm text-slate-400 max-w-2xl mx-auto">
            Synchronized extraction across three distinct communication channels for comprehensive pattern modeling.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Modality 1: Text */}
          <div className="rounded-2xl bg-slate-900/70 border border-slate-800 p-6 space-y-4 hover:border-emerald-500/40 transition-all shadow-lg flex flex-col">
            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-center">
              <MessageSquare className="w-6 h-6" />
            </div>
            <div>
              <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400">Modality 1</span>
              <h3 className="text-lg font-bold text-white">Text & Language Analysis</h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Multi-tiered NLP pipeline extracting syntactic, lexical, contextual, and affective structures from participant text.
            </p>
            <ul className="space-y-1.5 text-xs text-slate-400 border-t border-slate-800/80 pt-3 mt-auto font-mono">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                <span>TF-IDF & BERT Embeddings</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                <span>VADER & RoBERTa Sentiment</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                <span>Transformer Emotion Probabilities</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                <span>LDA Topics & Pronoun Statistics</span>
              </li>
            </ul>
          </div>

          {/* Modality 2: Voice */}
          <div className="rounded-2xl bg-slate-900/70 border border-slate-800 p-6 space-y-4 hover:border-indigo-500/40 transition-all shadow-lg flex flex-col">
            <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 flex items-center justify-center">
              <Mic className="w-6 h-6" />
            </div>
            <div>
              <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-400">Modality 2</span>
              <h3 className="text-lg font-bold text-white">Dual-Path Voice Analysis</h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Parallel speech-to-text semantic transcription (Path A) and low-level prosodic/acoustic feature extraction (Path B).
            </p>
            <ul className="space-y-1.5 text-xs text-slate-400 border-t border-slate-800/80 pt-3 mt-auto font-mono">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
                <span>Path A: Whisper ASR Transcription</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
                <span>Path B: Fundamental Pitch ($F_0$)</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
                <span>Energy (RMS) & Shimmer Dynamics</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
                <span>Speech Rate, Pauses & 13 MFCCs</span>
              </li>
            </ul>
          </div>

          {/* Modality 3: Communication */}
          <div className="rounded-2xl bg-slate-900/70 border border-slate-800 p-6 space-y-4 hover:border-teal-500/40 transition-all shadow-lg flex flex-col">
            <div className="w-12 h-12 rounded-xl bg-teal-500/10 border border-teal-500/30 text-teal-400 flex items-center justify-center">
              <BarChart3 className="w-6 h-6" />
            </div>
            <div>
              <span className="text-[11px] font-bold uppercase tracking-wider text-teal-400">Modality 3</span>
              <h3 className="text-lg font-bold text-white">Communication Dynamics</h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Temporal and structural metrics tracking turn-taking latency, interaction rhythm, and dyad reciprocity.
            </p>
            <ul className="space-y-1.5 text-xs text-slate-400 border-t border-slate-800/80 pt-3 mt-auto font-mono">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
                <span>Turn-Taking Response Latency</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
                <span>Message Cadence & Density</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
                <span>Sender/Receiver Balance Ratio</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
                <span>Conversation Initiation Tracking</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Personal Baseline & Fusion Paradigm */}
      <section className="max-w-6xl mx-auto">
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 sm:p-8 space-y-6">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
            <div>
              <div className="inline-flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
                <Binary className="w-4 h-4" />
                <span>Statistical Core</span>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-white">
                Personalized Baseline vs. Population Thresholds
              </h2>
            </div>
            <div className="px-3 py-1.5 rounded-lg bg-slate-800 text-xs font-mono text-slate-300 border border-slate-700">
              Formula: z = (x - μ_i) / (σ_i + ε)
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-xs sm:text-sm text-slate-300">
            <div className="space-y-3">
              <h3 className="font-semibold text-white text-base">Longitudinal Normalization</h3>
              <p className="leading-relaxed">
                Rather than comparing participant metrics against an arbitrary population norm, the engine establishes an individual historical baseline $(\mu_i, \sigma_i)$ computed over rolling longitudinal interaction sessions.
              </p>
              <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 font-mono text-xs space-y-1 text-slate-400">
                <p><span className="text-emerald-400">Stable Pattern:</span> |z| &lt; 1.0 (within typical range)</p>
                <p><span className="text-amber-400">Noticeable Change:</span> 1.0 ≤ |z| &lt; 2.0 (moderate deviation)</p>
                <p><span className="text-rose-400">Significant Change:</span> |z| ≥ 2.0 (high statistical deviation)</p>
              </div>
            </div>

            <div className="space-y-3">
              <h3 className="font-semibold text-white text-base">Explainable Pre-Screening Profile</h3>
              <p className="leading-relaxed">
                Feature vectors across all three modalities are concatenated via early fusion with explicit provenance tracking. The output delivers an interpretable multi-dimensional profile without black-box diagnostic assertions.
              </p>
              <div className="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-mono">Language Dimension:</span>
                  <span className="font-semibold text-emerald-400">Stable (|z| = 0.42)</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-mono">Acoustic Pitch ($F_0$):</span>
                  <span className="font-semibold text-amber-400">Noticeable Shift (|z| = 1.34)</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-mono">Turn Latency:</span>
                  <span className="font-semibold text-emerald-400">Stable (|z| = 0.28)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Roadmap Milestone Tracker */}
      <section className="max-w-6xl mx-auto space-y-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Clock className="w-5 h-5 text-emerald-400" />
          <span>Incremental Development Status</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-slate-900 border border-emerald-500/50 space-y-2 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-2 h-full bg-emerald-500"></div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-emerald-400 font-mono">MILESTONE 1</span>
              <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
                ACTIVE
              </span>
            </div>
            <h4 className="font-semibold text-white text-sm">Project Foundation</h4>
            <p className="text-xs text-slate-400">
              React + FastAPI scaffolding, health APIs, settings, and documentation suite.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 space-y-2 opacity-70">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 font-mono">MILESTONE 2</span>
              <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                PLANNED
              </span>
            </div>
            <h4 className="font-semibold text-slate-300 text-sm">Chat & Audio Transmission</h4>
            <p className="text-xs text-slate-400">
              Two-user WebSockets, in-browser audio recording, database persistence.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 space-y-2 opacity-70">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-400 font-mono">MILESTONE 3</span>
              <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                PLANNED
              </span>
            </div>
            <h4 className="font-semibold text-slate-300 text-sm">Unimodal NLP & Voice</h4>
            <p className="text-xs text-slate-400">
              TF-IDF, BERT, VADER, Whisper ASR, and Librosa acoustic feature extractors.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
