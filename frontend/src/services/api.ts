import axios from 'axios';
import {
  HealthResponse,
  SystemInfoResponse,
  RootEndpointResponse,
  ChatRoom,
  ConversationHistoryResponse,
  CommunicationMetrics,
  ConversationDetails,
  AnalyticalReportData,
  AnalyticalReportHistoryData,
  RoomAnalyticalReportsData
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
};

export const getSystemInfo = async (): Promise<SystemInfoResponse> => {
  const response = await apiClient.get<SystemInfoResponse>('/info');
  return response.data;
};

export const getRootInfo = async (): Promise<RootEndpointResponse> => {
  const rootUrl = API_BASE_URL.replace(/\/api\/v1\/?$/, '');
  const response = await axios.get<RootEndpointResponse>(rootUrl || '/');
  return response.data;
};

// --------------------------------------------------
// Milestone 2 & 3: Chat & Analysis Endpoints
// --------------------------------------------------

export const getActiveRooms = async (): Promise<ChatRoom[]> => {
  const response = await apiClient.get<ChatRoom[]>('/chat/rooms');
  return response.data;
};

export const createOrGetRoom = async (roomId: string, name?: string): Promise<ChatRoom> => {
  const response = await apiClient.post<ChatRoom>('/chat/rooms', {
    id: roomId,
    name: name || `Room ${roomId}`
  });
  return response.data;
};

export const getRoomHistory = async (roomId: string, limit = 100): Promise<ConversationHistoryResponse> => {
  const response = await apiClient.get<ConversationHistoryResponse>(`/chat/rooms/${roomId}/messages?limit=${limit}`);
  return response.data;
};

export const getActiveParticipants = async (roomId: string): Promise<{ room_id: string; participants: Array<{ user_id: string; display_name: string }>; count: number }> => {
  const response = await apiClient.get(`/chat/rooms/${roomId}/participants`);
  return response.data;
};

export const getRoomAnalysis = async (roomId: string, gapThreshold = 60.0): Promise<CommunicationMetrics> => {
  const response = await apiClient.get<CommunicationMetrics>(`/chat/rooms/${roomId}/analysis?gap_threshold=${gapThreshold}`);
  return response.data;
};

export const getConversationDetails = async (roomId: string): Promise<ConversationDetails> => {
  const response = await apiClient.get<ConversationDetails>(`/chat/conversations/${roomId}`);
  return response.data;
};

export const endConversation = async (roomId: string): Promise<ConversationDetails> => {
  const response = await apiClient.post<ConversationDetails>(`/chat/conversations/${roomId}/end`);
  return response.data;
};

export const uploadVoiceMessage = async (
  roomId: string,
  userId: string,
  audioBlob: Blob,
  durationSeconds?: number,
  receiverId?: string
): Promise<{ message: any; detail: string }> => {
  const formData = new FormData();
  formData.append('file', audioBlob, `recording_${Date.now()}.webm`);
  formData.append('user_id', userId);
  if (durationSeconds !== undefined) {
    formData.append('duration_seconds', durationSeconds.toString());
  }
  if (receiverId) {
    formData.append('receiver_id', receiverId);
  }

  const response = await apiClient.post<{ message: any; detail: string }>(
    `/chat/rooms/${roomId}/voice`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
};

export const transcribeVoiceMessage = async (
  messageId: string,
  language?: string
): Promise<{
  message_id: string;
  room_id: string;
  transcript_text?: string;
  transcription_status: string;
  transcription_model?: string;
  transcription_language?: string;
  transcription_error?: string;
  transcribed_at?: string;
  detail: string;
}> => {
  const query = language ? `?language=${encodeURIComponent(language)}` : '';
  const response = await apiClient.post(`/chat/messages/${messageId}/transcribe${query}`);
  return response.data;
};

export const analyzeAcousticFeatures = async (
  messageId: string
): Promise<any> => {
  const response = await apiClient.post(`/chat/messages/${messageId}/analyze-acoustic`);
  return response.data;
};

export const getAcousticFeatures = async (
  messageId: string
): Promise<any> => {
  const response = await apiClient.get(`/chat/messages/${messageId}/acoustic-features`);
  return response.data;
};

export const analyzeNLPFeatures = async (
  messageId: string
): Promise<any> => {
  const response = await apiClient.post(`/chat/messages/${messageId}/analyze-nlp`);
  return response.data;
};

export const getNLPFeatures = async (
  messageId: string
): Promise<any> => {
  const response = await apiClient.get(`/chat/messages/${messageId}/nlp-features`);
  return response.data;
};

export const fuseMultimodalFeatures = async (
  messageId: string
): Promise<any> => {
  const response = await apiClient.post(`/chat/messages/${messageId}/fuse`);
  return response.data;
};

export const getMultimodalFeatures = async (
  messageId: string
): Promise<any> => {
  const response = await apiClient.get(`/chat/messages/${messageId}/multimodal-features`);
  return response.data;
};

export const getRoomMultimodalFeatures = async (
  roomId: string
): Promise<{ room_id: string; total_analyzed: number; features: any[] }> => {
  const response = await apiClient.get(`/chat/rooms/${roomId}/multimodal-features`);
  return response.data;
};

// --------------------------------------------------
// Milestone 9: Personal Baseline & Temporal Trend Endpoints
// --------------------------------------------------

export const recomputeUserBaseline = async (
  userId: string,
  windowType: string = 'all_time'
): Promise<any> => {
  const response = await apiClient.post(`/chat/baselines/users/${userId}/recompute?window_type=${windowType}`);
  return response.data;
};

export const getUserBaseline = async (
  userId: string,
  windowType: string = 'all_time'
): Promise<any> => {
  const response = await apiClient.get(`/chat/baselines/users/${userId}?window_type=${windowType}`);
  return response.data;
};

export const evaluateMessageTemporal = async (
  messageId: string,
  windowType: string = 'all_time'
): Promise<any> => {
  const response = await apiClient.post(`/chat/temporal/messages/${messageId}/evaluate?window_type=${windowType}`);
  return response.data;
};

export const getMessageTemporal = async (
  messageId: string
): Promise<any> => {
  const response = await apiClient.get(`/chat/temporal/messages/${messageId}`);
  return response.data;
};

export const getUserTrajectory = async (
  userId: string,
  roomId?: string
): Promise<any> => {
  const url = roomId
    ? `/chat/temporal/users/${userId}/trajectory?room_id=${roomId}`
    : `/chat/temporal/users/${userId}/trajectory`;
  const response = await apiClient.get(url);
  return response.data;
};

export const getRoomTemporalTrends = async (
  roomId: string
): Promise<{ room_id: string; total_analyzed: number; records: any[] }> => {
  const response = await apiClient.get(`/chat/temporal/rooms/${roomId}`);
  return response.data;
};

// --------------------------------------------------
// Milestone 10: Final Analytical Report Endpoints
// --------------------------------------------------

export const generateAnalyticalReport = async (
  userId: string,
  params?: {
    room_id?: string;
    start_time?: string;
    end_time?: string;
    report_type?: string;
  }
): Promise<AnalyticalReportData> => {
  const response = await apiClient.post<AnalyticalReportData>(
    `/chat/analytical/users/${userId}/report`,
    params || {}
  );
  return response.data;
};

export const getLatestAnalyticalReport = async (
  userId: string,
  roomId?: string
): Promise<AnalyticalReportData> => {
  const url = roomId
    ? `/chat/analytical/users/${userId}/report?room_id=${roomId}`
    : `/chat/analytical/users/${userId}/report`;
  const response = await apiClient.get<AnalyticalReportData>(url);
  return response.data;
};

export const getAnalyticalReport = async (
  reportId: string
): Promise<AnalyticalReportData> => {
  const response = await apiClient.get<AnalyticalReportData>(`/chat/analytical/reports/${reportId}`);
  return response.data;
};

export const getAnalyticalReportHistory = async (
  userId: string,
  limit: number = 50
): Promise<AnalyticalReportHistoryData> => {
  const response = await apiClient.get<AnalyticalReportHistoryData>(
    `/chat/analytical/users/${userId}/reports?limit=${limit}`
  );
  return response.data;
};

export const getRoomAnalyticalReports = async (
  roomId: string,
  limit: number = 50
): Promise<RoomAnalyticalReportsData> => {
  const response = await apiClient.get<RoomAnalyticalReportsData>(
    `/chat/analytical/rooms/${roomId}?limit=${limit}`
  );
  return response.data;
};

export const exportAnalyticalReport = async (
  reportId: string,
  format: 'json' | 'markdown' | 'html' = 'json'
): Promise<any> => {
  const response = await apiClient.get(`/chat/analytical/reports/${reportId}/export?format=${format}`, {
    responseType: format === 'json' ? 'json' : 'text'
  });
  return response.data;
};

export default apiClient;


