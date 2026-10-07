export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  environment: string;
}

export interface SystemInfoResponse {
  project_title: string;
  system_type: string;
  disclaimer: string;
  version: string;
  modalities_supported: string[];
  baseline_paradigm: string;
}

export interface RootEndpointResponse {
  project: string;
  system_type: string;
  version: string;
  docs_url: string;
  health_check: string;
  disclaimer: string;
}

export interface AlgorithmRecord {
  id: string;
  algorithm: string;
  purpose: string;
  input: string;
  output: string;
  library: string;
  parameters: string;
  file: string;
  test: string;
  status: 'PLANNED' | 'IMPLEMENTED' | 'TESTED' | 'VALIDATED' | 'NEEDS REVISION';
}

// --------------------------------------------------
// Milestone 2: Chat & Real-Time Communication Types
// --------------------------------------------------

export interface ChatMessage {
  message_id: string;
  conversation_id: string;
  room_id: string;
  sender_id: string;
  sender_name?: string;
  receiver_id?: string | null;
  message_text: string;
  timestamp: string;
  message_type: 'text' | 'voice' | 'system';
  audio_url?: string;
  audio_duration_seconds?: number;
  audio_format?: string;
  audio_size_bytes?: number;
  transcript_text?: string;
  transcription_status?: 'pending' | 'processing' | 'completed' | 'failed';
  transcription_model?: string;
  transcription_language?: string;
  transcription_error?: string;
  transcribed_at?: string;
  has_acoustic_features?: boolean;
  has_nlp_features?: boolean;
  has_multimodal_features?: boolean;
  has_temporal_trends?: boolean;
}

export interface PitchFeaturesData {
  f0_mean_hz?: number | null;
  f0_std_hz?: number | null;
  f0_min_hz?: number | null;
  f0_max_hz?: number | null;
  f0_median_hz?: number | null;
  f0_range_hz?: number | null;
  voicing_fraction: number;
  f0_jitter_local?: number | null;
}

export interface EnergyFeaturesData {
  rms_mean: number;
  rms_std: number;
  rms_min: number;
  rms_max: number;
  rms_energy_range: number;
  shimmer_local?: number | null;
}

export interface SpeechRateFeaturesData {
  estimated_syllable_count: number;
  speech_duration_seconds: number;
  estimated_speaking_rate_syllables_per_sec: number;
  estimated_articulation_rate_syllables_per_sec?: number | null;
}

export interface PauseFeaturesData {
  pause_count: number;
  total_pause_duration_seconds: number;
  mean_pause_duration_seconds: number;
  max_pause_duration_seconds: number;
  pause_to_speech_ratio?: number | null;
}

export interface MFCCFeaturesData {
  mfcc_means: number[];
  mfcc_stds: number[];
  mfcc_dimension: number;
}

export interface SpectralFeaturesData {
  spectral_centroid_mean: number;
  spectral_centroid_std: number;
  spectral_bandwidth_mean: number;
  spectral_bandwidth_std: number;
  spectral_rolloff_mean: number;
  spectral_rolloff_std: number;
  spectral_flatness_mean: number;
  spectral_flatness_std: number;
  zcr_mean: number;
  zcr_std: number;
}

export interface AcousticFeatureData {
  message_id: string;
  room_id: string;
  sender_id: string;
  status: string;
  engine: string;
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  pitch_features: PitchFeaturesData;
  energy_features: EnergyFeaturesData;
  speech_rate_features: SpeechRateFeaturesData;
  pause_features: PauseFeaturesData;
  mfcc_features: MFCCFeaturesData;
  spectral_features: SpectralFeaturesData;
  feature_vector_version: string;
  raw_feature_vector: number[];
  has_voiced_frames: boolean;
  extraction_error?: string | null;
  extracted_at: string;
  detail?: string;
}

export interface LexicalFeaturesData {
  char_count: number;
  word_count: number;
  sentence_count: number;
  type_token_ratio_ttr: number;
  avg_word_length: number;
  first_person_singular_count: number;
  first_person_plural_count: number;
  second_person_count: number;
  third_person_count: number;
  pronoun_ratio: number;
}

export interface SentimentFeaturesData {
  vader_compound: number;
  vader_pos: number;
  vader_neu: number;
  vader_neg: number;
  sentiment_pos_prob: number;
  sentiment_neu_prob: number;
  sentiment_neg_prob: number;
  sentiment_polarity_score: number;
}

export interface EmotionFeaturesData {
  emotion_joy: number;
  emotion_sadness: number;
  emotion_anger: number;
  emotion_fear: number;
  emotion_surprise: number;
  emotion_disgust: number;
  emotion_neutral: number;
}

export interface TFIDFFeaturesData {
  tfidf_features: number[];
  feature_names: string[];
}

export interface EmbeddingFeaturesData {
  embedding_summary: number[];
  embedding_dimension: number;
}

export interface TopicFeaturesData {
  topic_0: number;
  topic_1: number;
  topic_2: number;
  topic_3: number;
  topic_4: number;
}

export interface NLPFeatureData {
  message_id: string;
  room_id: string;
  sender_id: string;
  source_type: 'text_message' | 'voice_transcript';
  status: string;
  engine: string;
  text_content: string;
  lexical_features: LexicalFeaturesData;
  sentiment_features: SentimentFeaturesData;
  emotion_features: EmotionFeaturesData;
  tfidf_features: TFIDFFeaturesData;
  embedding_features: EmbeddingFeaturesData;
  topic_features: TopicFeaturesData;
  feature_vector_version: string;
  raw_feature_vector: number[];
  extraction_error?: string | null;
  extracted_at: string;
  detail?: string;
}

// --------------------------------------------------
// Milestone 8: Multimodal Feature Fusion Types
// --------------------------------------------------

export interface CommunicationFeatureData {
  turn_index: number;
  sender_message_ratio: number;
  response_latency_seconds: number;
  is_communication_gap: number;
  conversation_duration_so_far_seconds: number;
  message_char_length: number;
  message_word_length: number;
  avg_char_length_session: number;
  avg_word_length_session: number;
  turn_switch_flag: number;
}

export interface MultimodalMaskData {
  has_communication_metrics: number;
  has_acoustic_features: number;
  has_nlp_features: number;
  is_voice_message: number;
  is_text_message: number;
}

export interface MultimodalFeatureData {
  message_id: string;
  room_id: string;
  sender_id: string;
  status: string;
  fusion_strategy: string;
  feature_vector_version: string;
  has_communication: boolean;
  has_acoustic: boolean;
  has_nlp: boolean;
  communication_features: CommunicationFeatureData;
  acoustic_features_summary?: {
    engine?: string;
    duration_seconds?: number;
    pitch_mean?: number | null;
    rms_mean?: number | null;
    pause_count?: number | null;
  } | null;
  nlp_features_summary?: {
    engine?: string;
    source_type?: string;
    vader_compound?: number | null;
    dominant_emotion?: string | null;
    word_count?: number | null;
  } | null;
  modality_mask: MultimodalMaskData;
  raw_fused_vector: number[];
  normalized_fused_vector: number[];
  fusion_error?: string | null;
  fused_at: string;
  detail?: string;
}

export interface ChatRoom {
  id: string;
  name?: string;
  status?: string;
  is_active: boolean;
  created_at: string;
  started_at?: string;
  ended_at?: string;
}

export interface ActiveParticipant {
  user_id: string;
  display_name: string;
}

export interface ConversationHistoryResponse {
  room_id: string;
  messages: ChatMessage[];
  total_messages: number;
}

export interface WebSocketEventPayload {
  event: 'join' | 'message' | 'leave' | 'user_joined' | 'user_left' | 'history' | 'error' | 'ping' | 'pong' | 'transcription_completed' | 'acoustic_analysis_completed' | 'nlp_analysis_completed' | 'multimodal_fusion_completed' | 'personal_baseline_updated' | 'temporal_deviation_computed' | 'analytical_report_generated';
  data: any;
  timestamp: string;
}

// --------------------------------------------------
// Milestone 9: Personal Baseline & Temporal Trend Types
// --------------------------------------------------

export interface TopDeviationItemData {
  index: number;
  name: string;
  modality: string;
  raw_zscore: number;
  clipped_zscore: number;
  absolute_zscore: number;
  raw_value: number;
  baseline_mean: number;
  baseline_std: number;
}

export interface PersonalBaselineData {
  user_id: string;
  window_type: string;
  feature_vector_version: string;
  sample_count: number;
  is_baseline_established: boolean;
  valid_dimensions_count: number;
  raw_mean_vector: number[];
  raw_std_vector: number[];
  normalized_mean_vector: number[];
  normalized_std_vector: number[];
  first_observed_at?: string | null;
  last_observed_at?: string | null;
  computed_at: string;
  detail?: string;
}

export interface TemporalTrendData {
  message_id: string;
  room_id: string;
  user_id: string;
  status: 'completed' | 'insufficient_history' | 'failed';
  feature_vector_version: string;
  sample_count_at_eval: number;
  is_baseline_established: boolean;
  baseline_snapshot_mean: number[];
  baseline_snapshot_std: number[];
  zscore_validity_mask: boolean[];
  valid_dimension_count: number;
  unavailable_dimension_count: number;
  modality_coverage: {
    has_communication: boolean;
    has_acoustic: boolean;
    has_nlp: boolean;
  };
  raw_zscore_vector: number[];
  normalized_zscore_vector: number[];
  clipped_zscore_vector: number[];
  mean_absolute_z: number;
  rms_z: number;
  standardized_euclidean_distance: number;
  top_deviations: TopDeviationItemData[];
  trend_slopes_summary: Record<string, any>;
  moving_averages_summary: Record<string, any>;
  rate_of_change_summary: Record<string, any>;
  evaluation_error?: string | null;
  evaluated_at: string;
  detail?: string;
}

export interface UserTrajectoryData {
  user_id: string;
  room_id?: string | null;
  total_points: number;
  trajectory: TemporalTrendData[];
}


// --------------------------------------------------
// Milestone 3: Communication Metrics & Session Types
// --------------------------------------------------

export interface ParticipantStats {
  user_id: string;
  display_name: string;
  message_count: number;
  message_percentage: number;
  average_message_length_chars: number;
  average_message_length_words: number;
  total_words: number;
}

export interface CommunicationMetrics {
  conversation_id: string;
  room_id: string;
  status: string;
  started_at?: string;
  ended_at?: string;
  participants: string[];
  participant_names: Record<string, string>;
  message_count: number;
  text_message_count: number;
  voice_message_count: number;
  messages_by_participant: Record<string, number>;
  message_ratio: Record<string, number>;
  average_message_length: number;
  average_message_length_words: number;
  min_message_length: number;
  max_message_length: number;
  conversation_duration_seconds: number;
  average_response_time_seconds?: number | null;
  min_response_time_seconds?: number | null;
  max_response_time_seconds?: number | null;
  number_of_communication_gaps: number;
  gap_threshold_seconds: number;
  participant_breakdown: ParticipantStats[];
}

export interface ConversationDetails {
  conversation_id: string;
  room_id: string;
  name?: string;
  status: string;
  is_active: boolean;
  created_at: string;
  started_at?: string;
  ended_at?: string;
  participant_count: number;
  message_count: number;
  participants: Array<{
    user_id: string;
    display_name: string;
    joined_at: string;
    last_seen_at: string;
  }>;
}

// --------------------------------------------------
// Milestone 10: Final Analytical Report Engine Types
// --------------------------------------------------

export interface ReportMetadataData {
  report_id: string;
  report_type: string;
  feature_vector_version: string;
  user_id: string;
  room_id?: string | null;
  observation_count: number;
  baseline_sample_count: number;
  baseline_established: boolean;
  generated_at: string;
}

export interface BaselineSummaryData {
  status: string;
  sample_count: number;
  window_type: string;
  feature_vector_version: string;
  is_baseline_established: boolean;
  valid_dimensions_count: number;
}

export interface ModalityCoverageItemData {
  modality: string;
  measured_count: number;
  total_count: number;
  percentage: number;
}

export interface ModalityCoverageSummaryData {
  communication: ModalityCoverageItemData;
  acoustic: ModalityCoverageItemData;
  nlp: ModalityCoverageItemData;
}

export interface DescriptiveMetricSummaryData {
  mean: number;
  median: number;
  minimum: number;
  maximum: number;
  std: number;
}

export interface DeviationSummaryData {
  mean_absolute_z: DescriptiveMetricSummaryData;
  rms_z: DescriptiveMetricSummaryData;
  standardized_euclidean_distance: DescriptiveMetricSummaryData;
  statistical_variation_category: string;
  category_description: string;
}

export interface FeatureShiftItemData {
  feature_index: number;
  feature_name: string;
  modality: string;
  valid_observations: number;
  mean_absolute_z: number;
  max_absolute_z: number;
  latest_signed_z: number;
  latest_absolute_z: number;
  baseline_mean: number;
  baseline_std: number;
}

export interface TrendFeatureItemData {
  feature_index: number;
  feature_name: string;
  modality: string;
  normalized_slope: number;
  raw_slope: number;
  direction: 'increasing' | 'decreasing' | 'stable';
  latest_sma3?: number | null;
  latest_sma5?: number | null;
  latest_ema3?: number | null;
  latest_ema5?: number | null;
  latest_roc?: number | null;
  valid_observation_count: number;
}

export interface TrendSummaryData {
  increasing_count: number;
  decreasing_count: number;
  stable_count: number;
  top_trends: TrendFeatureItemData[];
}

export interface LongitudinalTrajectoryPointData {
  message_id: string;
  timestamp: string;
  mean_absolute_z: number;
  rms_z: number;
  standardized_euclidean_distance: number;
  valid_dimension_count: number;
  top_deviation_feature?: string | null;
}

export interface TrajectorySummaryData {
  total_points: number;
  points: LongitudinalTrajectoryPointData[];
  timestamps: string[];
  mean_absolute_z_series: number[];
  rms_z_series: number[];
  standardized_euclidean_distance_series: number[];
}

export interface ReportPeriodData {
  period_start?: string | null;
  period_end?: string | null;
}

export interface AnalyticalReportData {
  metadata: ReportMetadataData;
  period: ReportPeriodData;
  baseline: BaselineSummaryData;
  modality_coverage: ModalityCoverageSummaryData;
  deviation_summary: DeviationSummaryData;
  feature_shifts: FeatureShiftItemData[];
  trend_summary: TrendSummaryData;
  trajectory_summary: TrajectorySummaryData;
  limitations: string[];
  disclaimer: string;
  detail?: string;
}

export interface AnalyticalReportHistoryData {
  user_id: string;
  total_reports: number;
  reports: AnalyticalReportData[];
}

export interface RoomAnalyticalReportsData {
  room_id: string;
  total_reports: number;
  reports: AnalyticalReportData[];
}

