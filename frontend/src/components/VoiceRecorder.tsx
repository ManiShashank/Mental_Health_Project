import React, { useState, useRef, useEffect } from 'react';
import {
  Mic,
  Square,
  Send,
  Trash2,
  AlertCircle,
  Loader2,
  CheckCircle2,
  Volume2
} from 'lucide-react';
import { uploadVoiceMessage } from '../services/api';

export type RecordingState =
  | 'idle'
  | 'requesting_permission'
  | 'recording'
  | 'stopped'
  | 'sending'
  | 'sent'
  | 'error';

interface Props {
  roomId: string;
  userId: string;
  onVoiceSent?: () => void;
  disabled?: boolean;
}

export const VoiceRecorder: React.FC<Props> = ({
  roomId,
  userId,
  onVoiceSent,
  disabled = false,
}) => {
  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Clean up preview URLs and stream on unmount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [audioPreviewUrl]);

  // Format seconds to mm:ss
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startRecording = async () => {
    setErrorMessage(null);
    setRecordingDuration(0);
    audioChunksRef.current = [];
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
      setAudioPreviewUrl(null);
    }
    setAudioBlob(null);

    setRecordingState('requesting_permission');

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('MediaDevices API not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Determine supported mime-type
      const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
        'audio/wav',
      ];
      let selectedMimeType = '';
      for (const mime of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mime)) {
          selectedMimeType = mime;
          break;
        }
      }

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: selectedMimeType || undefined,
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioType = selectedMimeType || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: audioType });
        setAudioBlob(blob);
        const url = URL.createObjectURL(blob);
        setAudioPreviewUrl(url);
        setRecordingState('stopped');

        // Stop mic tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(250); // Collect slice every 250ms

      setRecordingState('recording');

      // Start duration counter
      const startTime = Date.now();
      timerIntervalRef.current = window.setInterval(() => {
        setRecordingDuration((Date.now() - startTime) / 1000);
      }, 100);
    } catch (err: any) {
      console.error('Microphone access error:', err);
      setRecordingState('error');
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setErrorMessage('Microphone access denied. Please allow microphone permissions.');
      } else {
        setErrorMessage(err.message || 'Unable to access microphone.');
      }
    }
  };

  const stopRecording = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  };

  const cancelRecording = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
      setAudioPreviewUrl(null);
    }
    setAudioBlob(null);
    setRecordingDuration(0);
    setRecordingState('idle');
    setErrorMessage(null);
  };

  const sendVoiceMessage = async () => {
    if (!audioBlob) {
      setErrorMessage('No audio recorded to send.');
      return;
    }

    setRecordingState('sending');
    setErrorMessage(null);

    try {
      await uploadVoiceMessage(
        roomId,
        userId,
        audioBlob,
        Math.round(recordingDuration * 10) / 10
      );

      setRecordingState('sent');

      // Reset to idle after brief confirmation
      setTimeout(() => {
        cancelRecording();
        if (onVoiceSent) onVoiceSent();
      }, 800);
    } catch (err: any) {
      console.error('Failed to upload voice message:', err);
      setRecordingState('error');
      setErrorMessage(
        err.response?.data?.detail || err.message || 'Failed to send voice message.'
      );
    }
  };

  return (
    <div className="space-y-2">
      {/* Error Notice */}
      {errorMessage && (
        <div className="p-2.5 rounded-xl bg-rose-950/60 border border-rose-800/60 text-rose-300 text-xs flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-slate-400 hover:text-white text-xs font-bold px-1"
          >
            ✕
          </button>
        </div>
      )}

      {/* IDLE STATE: Micro-button trigger */}
      {recordingState === 'idle' && (
        <button
          type="button"
          onClick={startRecording}
          disabled={disabled}
          className="p-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-emerald-400 border border-slate-700 transition-all flex items-center gap-1.5 text-xs font-semibold cursor-pointer shadow-md"
          title="Record Voice Message"
        >
          <Mic className="w-4 h-4" />
          <span className="hidden sm:inline">Voice</span>
        </button>
      )}

      {/* REQUESTING PERMISSION STATE */}
      {recordingState === 'requesting_permission' && (
        <div className="p-2.5 rounded-xl bg-slate-900 border border-indigo-500/40 text-indigo-300 text-xs flex items-center gap-2 animate-pulse">
          <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
          <span>Requesting microphone permission...</span>
          <button
            type="button"
            onClick={cancelRecording}
            className="ml-auto text-slate-400 hover:text-white text-xs px-2 py-0.5 rounded bg-slate-800"
          >
            Cancel
          </button>
        </div>
      )}

      {/* RECORDING STATE */}
      {recordingState === 'recording' && (
        <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 flex items-center justify-between gap-3 shadow-lg">
          <div className="flex items-center gap-2.5">
            <span className="w-3 h-3 rounded-full bg-rose-500 animate-ping flex-shrink-0" />
            <span className="text-xs font-semibold text-rose-300 font-mono">
              Recording... {formatDuration(recordingDuration)}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={stopRecording}
              className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-md cursor-pointer transition-all"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Stop</span>
            </button>
            <button
              type="button"
              onClick={cancelRecording}
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-rose-300 transition-colors"
              title="Cancel recording"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* STOPPED / PREVIEW STATE */}
      {recordingState === 'stopped' && audioPreviewUrl && (
        <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-2.5 shadow-xl">
          <div className="flex items-center justify-between text-xs text-slate-300">
            <div className="flex items-center gap-1.5 font-medium text-emerald-400">
              <Volume2 className="w-4 h-4" />
              <span>Audio Preview ({formatDuration(recordingDuration)})</span>
            </div>
            <span className="text-[10px] text-slate-500 font-mono">Ready to transmit</span>
          </div>

          {/* Native HTML5 Audio Preview */}
          <audio controls src={audioPreviewUrl} className="w-full h-8 rounded-lg" />

          <div className="flex items-center justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={cancelRecording}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-rose-950/60 hover:text-rose-300 text-slate-300 text-xs font-medium border border-slate-700 flex items-center gap-1 cursor-pointer transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Delete</span>
            </button>
            <button
              type="button"
              onClick={sendVoiceMessage}
              className="px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-lg shadow-emerald-950/50 cursor-pointer transition-all"
            >
              <Send className="w-3.5 h-3.5" />
              <span>Send Voice</span>
            </button>
          </div>
        </div>
      )}

      {/* SENDING STATE */}
      {recordingState === 'sending' && (
        <div className="p-2.5 rounded-xl bg-slate-900 border border-emerald-500/40 text-emerald-300 text-xs flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
          <span>Uploading and broadcasting voice message...</span>
        </div>
      )}

      {/* SENT CONFIRMATION STATE */}
      {recordingState === 'sent' && (
        <div className="p-2.5 rounded-xl bg-emerald-950/50 border border-emerald-800/60 text-emerald-300 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>Voice message transmitted successfully!</span>
        </div>
      )}

      {/* ERROR FALLBACK RETRY */}
      {recordingState === 'error' && (
        <button
          type="button"
          onClick={cancelRecording}
          className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium"
        >
          Reset Recorder
        </button>
      )}
    </div>
  );
};
