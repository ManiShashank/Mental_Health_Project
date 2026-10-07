import React, { useState } from 'react';
import {
  BookOpen,
  Layers,
  Cpu,
  GitBranch,
  FileCode
} from 'lucide-react';
import { AlgorithmRecord } from '../types';

const ALGORITHM_DATA: AlgorithmRecord[] = [
  {
    id: 'NLP-01',
    algorithm: 'Text Preprocessing & Normalization',
    purpose: 'Tokenize, lemmatize, filter stopwords, clean text',
    input: 'Raw text string',
    output: 'Clean token sequence',
    library: 'spaCy / NLTK',
    parameters: 'lower=True, strip_punct=True',
    file: 'backend/app/analysis/nlp/preprocessor.py',
    test: 'backend/tests/test_nlp_preprocessor.py',
    status: 'PLANNED',
  },
  {
    id: 'NLP-02',
    algorithm: 'TF-IDF Lexical Feature Extraction',
    purpose: 'Lexical frequency-inverse document frequency vectorization',
    input: 'Preprocessed tokens',
    output: 'Sparse lexical vector (1 x V)',
    library: 'scikit-learn TfidfVectorizer',
    parameters: 'max_features=500, ngram_range=(1,2)',
    file: 'backend/app/analysis/nlp/tfidf_extractor.py',
    test: 'backend/tests/test_tfidf.py',
    status: 'PLANNED',
  },
  {
    id: 'NLP-03',
    algorithm: 'BERT Contextual Embeddings',
    purpose: 'Dense semantic contextual representation',
    input: 'Raw / Tokenized text',
    output: '768-d dense embedding vector',
    library: 'transformers (distilbert-base-uncased)',
    parameters: 'max_length=128, pooling=mean',
    file: 'backend/app/analysis/nlp/bert_extractor.py',
    test: 'backend/tests/test_bert.py',
    status: 'PLANNED',
  },
  {
    id: 'NLP-04',
    algorithm: 'VADER Sentiment Analysis',
    purpose: 'Rule-based lexical baseline sentiment scoring',
    input: 'Raw message string',
    output: 'Compound, pos, neu, neg scores',
    library: 'nltk.sentiment.vader',
    parameters: 'Default lexicon weights',
    file: 'backend/app/analysis/nlp/vader_sentiment.py',
    test: 'backend/tests/test_vader.py',
    status: 'PLANNED',
  },
  {
    id: 'NLP-05',
    algorithm: 'Transformer Sentiment Analysis',
    purpose: 'Deep contextual sentiment classification',
    input: 'Raw message string',
    output: 'Probabilities (Positive, Neutral, Negative)',
    library: 'cardiffnlp/twitter-roberta-base-sentiment',
    parameters: 'return_all_scores=True',
    file: 'backend/app/analysis/nlp/transformer_sentiment.py',
    test: 'backend/tests/test_transformer_sentiment.py',
    status: 'PLANNED',
  },
  {
    id: 'NLP-06',
    algorithm: 'Transformer Emotion Classification',
    purpose: 'Multi-class affective state probability distribution',
    input: 'Raw message string',
    output: 'Emotion probability distribution (6-7 classes)',
    library: 'bhadresh-psav/roberta-base-go_emotions',
    parameters: 'return_all_scores=True',
    file: 'backend/app/analysis/nlp/emotion_classifier.py',
    test: 'backend/tests/test_emotion.py',
    status: 'PLANNED',
  },
  {
    id: 'NLP-07',
    algorithm: 'LDA Topic Extraction',
    purpose: 'Unsupervised thematic topic distribution',
    input: 'Preprocessed token corpus',
    output: 'Topic distribution vector (1 x K)',
    library: 'scikit-learn LatentDirichletAllocation',
    parameters: 'n_components=5, alpha=0.1',
    file: 'backend/app/analysis/nlp/lda_topics.py',
    test: 'backend/tests/test_lda.py',
    status: 'PLANNED',
  },
  {
    id: 'NLP-08',
    algorithm: 'Syntactic & Lexical Statistics',
    purpose: 'Word count, sentence count, TTR diversity, pronouns, POS',
    input: 'Raw text & POS tags',
    output: 'Dict of numeric linguistic metrics',
    library: 'spaCy / NLTK',
    parameters: 'pronoun_types=[1st_sing, 1st_plur, 2nd, 3rd]',
    file: 'backend/app/analysis/nlp/linguistic_stats.py',
    test: 'backend/tests/test_linguistic_stats.py',
    status: 'PLANNED',
  },
  {
    id: 'AUD-01',
    algorithm: 'Whisper ASR (Path A)',
    purpose: 'Automatic Speech Recognition converting audio to text',
    input: 'Audio stream / file (WAV/FLAC)',
    output: 'Transcribed text string + word timestamps',
    library: 'openai-whisper (base/small)',
    parameters: 'fp16=False, language=en',
    file: 'backend/app/analysis/voice/speech_to_text.py',
    test: 'backend/tests/test_asr.py',
    status: 'PLANNED',
  },
  {
    id: 'AUD-02',
    algorithm: 'Fundamental Frequency (F0 Pitch)',
    purpose: 'Measure vocal pitch distribution, mean, std, jitter',
    input: 'Audio signal (16 kHz)',
    output: 'Mean F0, Std F0, Min, Max, Range',
    library: 'librosa.pyin',
    parameters: 'fmin=65, fmax=500, sr=16000',
    file: 'backend/app/analysis/voice/pitch_extractor.py',
    test: 'backend/tests/test_pitch.py',
    status: 'PLANNED',
  },
  {
    id: 'AUD-03',
    algorithm: 'Energy / RMS & Shimmer',
    purpose: 'Compute acoustic energy dynamics and loudness variations',
    input: 'Audio signal',
    output: 'Mean RMS, RMS Variance, Shimmer',
    library: 'librosa.feature.rms',
    parameters: 'frame_length=2048, hop_length=512',
    file: 'backend/app/analysis/voice/energy_extractor.py',
    test: 'backend/tests/test_energy.py',
    status: 'PLANNED',
  },
  {
    id: 'AUD-04',
    algorithm: 'Speech Rate & Pause Dynamics',
    purpose: 'Quantify speaking tempo, total pause duration, voiced ratio',
    input: 'Audio signal & VAD segments',
    output: 'Syllables/sec, Pause Count, Pause Duration Ratio',
    library: 'librosa / webrtcvad',
    parameters: 'silence_threshold_db=-30',
    file: 'backend/app/analysis/voice/speech_rate_extractor.py',
    test: 'backend/tests/test_speech_rate.py',
    status: 'PLANNED',
  },
  {
    id: 'AUD-05',
    algorithm: 'MFCCs (Cepstral Representation)',
    purpose: 'Spectral envelope and timbre feature extraction',
    input: 'Audio signal',
    output: '13 MFCC means & standard deviations (26-d)',
    library: 'librosa.feature.mfcc',
    parameters: 'n_mfcc=13, n_mels=40',
    file: 'backend/app/analysis/voice/mfcc_extractor.py',
    test: 'backend/tests/test_mfcc.py',
    status: 'PLANNED',
  },
  {
    id: 'COM-01',
    algorithm: 'Interaction Turn-Taking Latency',
    purpose: 'Measure response delay between dyad participants',
    input: 'Timestamped message logs',
    output: 'Mean response time, response delay variance',
    library: 'Custom Python Engine',
    parameters: 'Timestamp delta in seconds',
    file: 'backend/app/analysis/communication/turn_dynamics.py',
    test: 'backend/tests/test_turn_dynamics.py',
    status: 'PLANNED',
  },
  {
    id: 'BAS-01',
    algorithm: 'Longitudinal Personal Baseline',
    purpose: 'Compute individual mean mu and std sigma per feature',
    input: 'Participant history (N >= 3 sessions)',
    output: 'Baseline vector (mu, sigma)',
    library: 'Custom Statistical Engine',
    parameters: 'Rolling window: N=10, min_samples: 3',
    file: 'backend/app/analysis/baseline/personal_baseline.py',
    test: 'backend/tests/test_personal_baseline.py',
    status: 'PLANNED',
  },
  {
    id: 'BAS-02',
    algorithm: 'Z-Score Deviation Engine',
    purpose: 'Standardize current session features against personal baseline',
    input: 'Current feature vector & Baseline (mu, sigma)',
    output: 'Dimensional z-scores vector',
    library: 'Custom Statistical Engine',
    parameters: 'epsilon = 1e-6',
    file: 'backend/app/analysis/baseline/zscore_engine.py',
    test: 'backend/tests/test_zscore.py',
    status: 'PLANNED',
  },
  {
    id: 'PRF-01',
    algorithm: 'Explainable Pre-Screening Profiler',
    purpose: 'Generate rule-based dimensional stability profile',
    input: 'Dimensional z-scores & fused vectors',
    output: 'Category statuses (Stable / Noticeable / Significant)',
    library: 'Custom Rule Engine',
    parameters: 'Stable (<1.0), Noticeable (1.0-2.0), Significant (>=2.0)',
    file: 'backend/app/analysis/profile/prescreening_profiler.py',
    test: 'backend/tests/test_prescreening.py',
    status: 'PLANNED',
  },
];

export const Documentation: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'spec' | 'registry' | 'architecture' | 'roadmap'>('registry');
  const [filterText, setFilterText] = useState<string>('');

  const filteredAlgorithms = ALGORITHM_DATA.filter((alg) =>
    alg.algorithm.toLowerCase().includes(filterText.toLowerCase()) ||
    alg.id.toLowerCase().includes(filterText.toLowerCase()) ||
    alg.library.toLowerCase().includes(filterText.toLowerCase()) ||
    alg.purpose.toLowerCase().includes(filterText.toLowerCase())
  );

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-12">
      {/* Header */}
      <div className="border-b border-slate-800 pb-6">
        <div className="inline-flex items-center gap-2 text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">
          <BookOpen className="w-4 h-4 text-emerald-400" />
          <span>Documentation Suite</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-white">System Documentation & Registry</h1>
        <p className="text-sm text-slate-400 mt-1">
          Traceable records of academic specifications, algorithm registries, architectures, and verification guidelines.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-3">
        <button
          onClick={() => setActiveTab('registry')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
            activeTab === 'registry'
              ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-950/50'
              : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Algorithm Registry ({ALGORITHM_DATA.length})</span>
        </button>

        <button
          onClick={() => setActiveTab('spec')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
            activeTab === 'spec'
              ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-950/50'
              : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
          }`}
        >
          <FileCode className="w-4 h-4" />
          <span>Project Specification</span>
        </button>

        <button
          onClick={() => setActiveTab('architecture')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
            activeTab === 'architecture'
              ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-950/50'
              : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
          }`}
        >
          <Cpu className="w-4 h-4" />
          <span>System Architecture</span>
        </button>

        <button
          onClick={() => setActiveTab('roadmap')}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all cursor-pointer ${
            activeTab === 'roadmap'
              ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-950/50'
              : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
          }`}
        >
          <GitBranch className="w-4 h-4" />
          <span>Development Roadmap</span>
        </button>
      </div>

      {/* Tab: Algorithm Registry */}
      {activeTab === 'registry' && (
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
            <div className="relative flex-1 max-w-md">
              <input
                type="text"
                placeholder="Search algorithms, libraries, or parameters..."
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                className="w-full px-4 py-2 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500 font-sans"
              />
            </div>
            <div className="text-xs text-slate-400 font-mono">
              Showing {filteredAlgorithms.length} of {ALGORITHM_DATA.length} registered algorithms
            </div>
          </div>

          <div className="rounded-2xl bg-slate-900/80 border border-slate-800 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-950/80 text-slate-400 font-mono">
                    <th className="p-3 font-semibold">ID</th>
                    <th className="p-3 font-semibold">ALGORITHM</th>
                    <th className="p-3 font-semibold">PURPOSE</th>
                    <th className="p-3 font-semibold">LIBRARY / MODEL</th>
                    <th className="p-3 font-semibold">PARAMETERS</th>
                    <th className="p-3 font-semibold">STATUS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-sans text-slate-300">
                  {filteredAlgorithms.map((alg) => (
                    <tr key={alg.id} className="hover:bg-slate-800/30 transition-colors">
                      <td className="p-3 font-mono font-bold text-emerald-400">{alg.id}</td>
                      <td className="p-3 font-semibold text-white">{alg.algorithm}</td>
                      <td className="p-3 text-slate-400 max-w-xs">{alg.purpose}</td>
                      <td className="p-3 font-mono text-[11px] text-indigo-300">{alg.library}</td>
                      <td className="p-3 font-mono text-[11px] text-slate-400">{alg.parameters}</td>
                      <td className="p-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider font-mono ${
                          alg.status === 'VALIDATED'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                            : alg.status === 'TESTED'
                            ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30'
                            : alg.status === 'IMPLEMENTED'
                            ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                            : 'bg-slate-800 text-slate-400 border border-slate-700'
                        }`}>
                          {alg.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab: Project Specification */}
      {activeTab === 'spec' && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 sm:p-8 space-y-6 text-xs sm:text-sm text-slate-300">
          <h2 className="text-xl font-bold text-white">Project Specification & Mathematical Formulation</h2>
          
          <div className="space-y-4">
            <h3 className="text-base font-semibold text-emerald-400">1. Personalized Longitudinal Baseline Formula</h3>
            <p className="leading-relaxed">
              For any numeric feature vector x<sub>i,t</sub> observed for participant i during interaction session t, the longitudinal baseline mean μ<sub>i</sub> and standard deviation σ<sub>i</sub> are computed over prior historical sessions:
            </p>
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-emerald-300 text-xs">
              μ_i = (1 / N) * Σ x_i,k<br/>
              σ_i = sqrt( (1 / (N - 1)) * Σ (x_i,k - μ_i)^2 )<br/>
              z_i,t = (x_i,t - μ_i) / (σ_i + ε)  [where ε = 1e-6]
            </div>
          </div>

          <div className="space-y-3 pt-4 border-t border-slate-800">
            <h3 className="text-base font-semibold text-emerald-400">2. Tri-Modal Feature Fusion</h3>
            <p className="leading-relaxed">
              Features from text (TF-IDF, BERT, sentiment, emotions, linguistic stats), voice (Whisper transcript NLP + Librosa F0, RMS, speech rate, MFCCs), and communication dynamics (turn latency, cadence, balance ratio) are normalized and concatenated into a standardized feature vector.
            </p>
          </div>
        </div>
      )}

      {/* Tab: System Architecture */}
      {activeTab === 'architecture' && (
        <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 sm:p-8 space-y-6 text-xs sm:text-sm text-slate-300">
          <h2 className="text-xl font-bold text-white">System Architecture & Component Decoupling</h2>
          <p className="leading-relaxed">
            The platform follows a strictly modular architecture separating the presentation layer (React SPA), API and broker layer (FastAPI & WebSockets), analytical processing pipelines (pure functional ML/NLP engines), and persistence layers.
          </p>

          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-indigo-300 whitespace-pre overflow-x-auto">
{`+-------------------------------------------------------------+
|                     React 18 Frontend                       |
|   Dyadic Chat UI  |  Analytics Dashboard  |  Profile View   |
+------------------------------+------------------------------+
                               | (WebSocket & REST /api/v1)
                               v
+-------------------------------------------------------------+
|                  FastAPI Backend Routing                    |
|   Connection Broker  |  Pydantic Schemas  |  Config Engine  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 Analytical Processing Core                  |
|   NLP Engine (BERT/VADER)  |  Voice Engine (Whisper/Librosa)|
|   Communication Dynamics   |  Personal Baseline (Z-Score)   |
|   Early Multimodal Fusion  |  Explainable Pre-Screening     |
+-------------------------------------------------------------+`}
          </div>
        </div>
      )}

      {/* Tab: Development Roadmap */}
      {activeTab === 'roadmap' && (
        <div className="space-y-4">
          <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                <span>Milestone 1: Project Foundation</span>
              </h3>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                ACTIVE & READY FOR AUDIT
              </span>
            </div>
            <p className="text-xs text-slate-300">
              Scaffolding, React + Vite frontend, FastAPI backend with health endpoints, centralized Pydantic settings, Pytest test suite, and full documentation architecture.
            </p>
          </div>

          <div className="rounded-2xl bg-slate-900/40 border border-slate-800 p-6 space-y-4 opacity-75">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-slate-300 flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-slate-600"></span>
                <span>Milestone 2: Communication Infrastructure & Persistence</span>
              </h3>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-slate-800 text-slate-400">
                UPCOMING
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Two-user WebSocket messaging, in-browser audio recording, database persistence (PostgreSQL/SQLite), and session state management.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
