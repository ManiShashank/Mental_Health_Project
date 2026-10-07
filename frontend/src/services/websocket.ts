import { WebSocketEventPayload, ChatMessage, ActiveParticipant } from '../types';

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

export interface ChatWebSocketHandlers {
  onMessage?: (message: ChatMessage) => void;
  onHistory?: (messages: ChatMessage[], participants: ActiveParticipant[]) => void;
  onUserJoined?: (data: { user_id: string; display_name: string; system_message: string; participants: ActiveParticipant[] }) => void;
  onUserLeft?: (data: { user_id: string; display_name: string; system_message: string; participants: ActiveParticipant[] }) => void;
  onTranscriptionCompleted?: (data: { message_id: string; room_id: string; transcript_text: string; transcription_status: string; transcription_model: string; transcription_language?: string; transcribed_at?: string }) => void;
  onAcousticAnalysisCompleted?: (data: { message_id: string; room_id: string; sender_id: string; status: string; engine: string; duration_seconds: number; has_voiced_frames: boolean; feature_vector_version: string; extracted_at: string }) => void;
  onNLPAnalysisCompleted?: (data: { message_id: string; room_id: string; sender_id: string; source_type: string; status: string; engine: string; word_count: number; sentiment_polarity: number; feature_vector_version: string; extracted_at: string }) => void;
  onMultimodalFusionCompleted?: (data: { message_id: string; room_id: string; sender_id: string; fusion_strategy: string; feature_vector_version: string; raw_feature_vector_length: number; has_communication: boolean; has_acoustic: boolean; has_nlp: boolean; fused_at: string }) => void;
  onPersonalBaselineUpdated?: (data: { user_id: string; sample_count: number; is_baseline_established: boolean; valid_dimensions_count: number; evaluated_at: string }) => void;
  onTemporalDeviationComputed?: (data: { message_id: string; room_id: string; user_id: string; status: string; sample_count_at_eval: number; is_baseline_established: boolean; valid_dimension_count: number; mean_absolute_z: number; rms_z: number; standardized_euclidean_distance: number; top_deviations: any[]; evaluated_at: string }) => void;
  onAnalyticalReportGenerated?: (data: { report_id: string; user_id: string; room_id?: string; observation_count: number; baseline_established: boolean; generated_at: string }) => void;
  onError?: (errorDetail: string) => void;
  onStatusChange?: (status: ConnectionStatus) => void;
}

export class ChatWebSocketClient {
  private ws: WebSocket | null = null;
  private roomId: string;
  private userId: string;
  private displayName: string;
  private handlers: ChatWebSocketHandlers;
  private shouldReconnect: boolean = true;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectTimer: any = null;

  constructor(roomId: string, userId: string, displayName: string, handlers: ChatWebSocketHandlers) {
    this.roomId = roomId;
    this.userId = userId;
    this.displayName = displayName;
    this.handlers = handlers;
  }

  public connect(): void {
    this.shouldReconnect = true;
    this.handlers.onStatusChange?.(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname === 'localhost' ? '127.0.0.1:8000' : `${window.location.hostname}:8000`;
    const wsUrl = `${protocol}//${host}/api/v1/chat/ws/${encodeURIComponent(this.roomId)}/${encodeURIComponent(this.userId)}?display_name=${encodeURIComponent(this.displayName)}`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.handlers.onStatusChange?.('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const payload: WebSocketEventPayload = JSON.parse(event.data);
          this.handleEvent(payload);
        } catch (err) {
          console.error('Failed to parse incoming WebSocket message:', err);
        }
      };

      this.ws.onerror = (err) => {
        console.warn('WebSocket encountered error:', err);
        this.handlers.onError?.('WebSocket connection error occurred.');
      };

      this.ws.onclose = () => {
        this.handlers.onStatusChange?.('disconnected');
        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++;
          const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
          this.reconnectTimer = setTimeout(() => this.connect(), delay);
        }
      };
    } catch (err: any) {
      console.error('Failed to create WebSocket:', err);
      this.handlers.onStatusChange?.('disconnected');
      this.handlers.onError?.(err.message || 'Failed to establish WebSocket connection');
    }
  }

  private handleEvent(payload: WebSocketEventPayload): void {
    switch (payload.event) {
      case 'history':
        this.handlers.onHistory?.(payload.data.messages || [], payload.data.participants || []);
        break;
      case 'message':
        this.handlers.onMessage?.(payload.data);
        break;
      case 'user_joined':
        this.handlers.onUserJoined?.(payload.data);
        break;
      case 'user_left':
        this.handlers.onUserLeft?.(payload.data);
        break;
      case 'transcription_completed':
        this.handlers.onTranscriptionCompleted?.(payload.data);
        break;
      case 'acoustic_analysis_completed':
        this.handlers.onAcousticAnalysisCompleted?.(payload.data);
        break;
      case 'nlp_analysis_completed':
        this.handlers.onNLPAnalysisCompleted?.(payload.data);
        break;
      case 'multimodal_fusion_completed':
        this.handlers.onMultimodalFusionCompleted?.(payload.data);
        break;
      case 'personal_baseline_updated':
        this.handlers.onPersonalBaselineUpdated?.(payload.data);
        break;
      case 'temporal_deviation_computed':
        this.handlers.onTemporalDeviationComputed?.(payload.data);
        break;
      case 'analytical_report_generated':
        this.handlers.onAnalyticalReportGenerated?.(payload.data);
        break;
      case 'error':
        this.handlers.onError?.(payload.data.detail || 'Unknown server error');
        break;
      case 'pong':
        break;
      default:
        break;
    }
  }

  public sendMessage(text: string, receiverId?: string): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.handlers.onError?.('Cannot send message: WebSocket is disconnected.');
      return false;
    }

    const payload = {
      event: 'message',
      data: {
        message_text: text,
        receiver_id: receiverId || null,
        message_type: 'text'
      }
    };

    try {
      this.ws.send(JSON.stringify(payload));
      return true;
    } catch (err: any) {
      this.handlers.onError?.(`Send failed: ${err.message}`);
      return false;
    }
  }

  public disconnect(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    if (this.ws) {
      try {
        if (this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ event: 'leave', data: {} }));
        }
        this.ws.close();
      } catch (err) {
        console.warn('Error during disconnect:', err);
      }
      this.ws = null;
    }
    this.handlers.onStatusChange?.('disconnected');
  }
}
