import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { useActiveUser } from "../state/ActiveUserContext";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://127.0.0.1:8000";

type VoiceMsg =
  | { type: "transcript"; text: string; is_final: boolean }
  | { type: "token"; content: string }
  | { type: "audio"; content: string; sentence?: string }
  | { type: "pause_mic" }
  | { type: "resume_mic" }
  | { type: "done"; full_response: string }
  | { type: "error"; message: string };

export const TestPage = () => {
  const { activeUser } = useActiveUser();

  const [status, setStatus] = useState<string>("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [voice, setVoice] = useState<string>("dave");
  const [systemPrompt, setSystemPrompt] = useState<string>("You are a helpful voice assistant. Be concise.");

  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const isRecordingRef = useRef(false);
  const isPausedRef = useRef(false);

  const [finalTranscript, setFinalTranscript] = useState<string>("");
  const [assistantText, setAssistantText] = useState<string>("");

  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const audioQueueRef = useRef<string[]>([]);
  const isPlayingRef = useRef(false);

  const wsUrl = useMemo(() => {
    const u = new URL(API_BASE);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    u.pathname = "/ws/voice";
    return u.toString();
  }, []);

  const playNextAudio = async () => {
    if (isPlayingRef.current) return;
    const next = audioQueueRef.current.shift();
    if (!next) return;

    isPlayingRef.current = true;
    try {
      const audio = new Audio(`data:audio/wav;base64,${next}`);
      await audio.play();
      audio.onended = () => {
        isPlayingRef.current = false;
        playNextAudio();
      };
    } catch {
      isPlayingRef.current = false;
    }
  };

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const ps = await api.getPersonalities(true);
        const selectedId = activeUser?.current_personality_id;
        const selected = ps.find((p: any) => p.id === selectedId);
        if (!cancelled && selected) {
          setVoice(selected.voice_id || "dave");
          setSystemPrompt(selected.prompt || "You are a helpful voice assistant. Be concise.");
        }
      } catch {
        // ignore
      }
    };

    load();
  }, [activeUser?.current_personality_id]);

  const stopRecording = () => {
    setIsRecording(false);
    isRecordingRef.current = false;
    isPausedRef.current = false;
    setIsPaused(false);

    try {
      processorRef.current?.disconnect();
    } catch {
      // ignore
    }
    try {
      sourceRef.current?.disconnect();
    } catch {
      // ignore
    }
    try {
      audioCtxRef.current?.close();
    } catch {
      // ignore
    }
    audioCtxRef.current = null;
    processorRef.current = null;
    sourceRef.current = null;

    try {
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {
      // ignore
    }
    mediaStreamRef.current = null;
  };

  const startRecording = async () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError("Voice WebSocket is not connected");
      return;
    }

    setError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone access requires a secure context (HTTPS) or localhost");
      return;
    }

    const mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    mediaStreamRef.current = mediaStream;

    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;

    const source = audioCtx.createMediaStreamSource(mediaStream);
    sourceRef.current = source;

    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processorRef.current = processor;

    const actualRate = audioCtx.sampleRate;

    processor.onaudioprocess = (e) => {
      const socket = wsRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      if (!isRecordingRef.current) return;
      if (isPausedRef.current) return;

      const input = e.inputBuffer.getChannelData(0);

      const ratio = 24000 / actualRate;
      const outputLen = Math.round(input.length * ratio);
      const resampled = new Float32Array(outputLen);
      for (let i = 0; i < outputLen; i++) {
        const srcIdx = i / ratio;
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(idx0 + 1, input.length - 1);
        const frac = srcIdx - idx0;
        resampled[i] = input[idx0] * (1 - frac) + input[idx1] * frac;
      }

      const pcm = new Int16Array(resampled.length);
      for (let i = 0; i < resampled.length; i++) {
        pcm[i] = Math.max(-32768, Math.min(32767, resampled[i] * 32768));
      }

      socket.send(pcm.buffer);
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);
    setIsRecording(true);
    isRecordingRef.current = true;
  };

  const toggleRecording = async () => {
    if (isRecording) {
      stopRecording();
    } else {
      await startRecording();
    }
  };

  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      try {
        ws.send(JSON.stringify({ voice, system_prompt: systemPrompt }));
      } catch {
        // ignore
      }
    };
    ws.onclose = () => {
      setStatus("disconnected");
      stopRecording();
    };
    ws.onerror = () => setStatus("error");

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as VoiceMsg;

        if (msg.type === "transcript" && msg.is_final) {
          setFinalTranscript(msg.text || "");
          setAssistantText("");
        } else if (msg.type === "token") {
          setAssistantText((t) => t + (msg.content || ""));
        } else if (msg.type === "audio") {
          if (msg.content) {
            audioQueueRef.current.push(msg.content);
            playNextAudio();
          }
        } else if (msg.type === "pause_mic") {
          isPausedRef.current = true;
          setIsPaused(true);
        } else if (msg.type === "resume_mic") {
          isPausedRef.current = false;
          setIsPaused(false);
        } else if (msg.type === "error") {
          setError(msg.message || "Unknown error");
        }
      } catch {
        // ignore
      }
    };

    return () => {
      try {
        ws.close();
      } catch {
        // ignore
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsUrl]);

  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ voice, system_prompt: systemPrompt }));
    } catch {
      // ignore
    }
  }, [voice, systemPrompt]);

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-black">TEST</h2>
        <div className="font-mono text-xs text-gray-500">{status}</div>
      </div>

      {error && <div className="retro-card font-mono text-sm mb-4">{error}</div>}

      <div className="retro-card mb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="font-bold uppercase text-xs mb-1">Voice</div>
            <input className="retro-input w-full" value={voice} onChange={(e) => setVoice(e.target.value)} />
          </div>
          <div>
            <div className="font-bold uppercase text-xs mb-1">System prompt</div>
            <input
              className="retro-input w-full"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
          </div>
        </div>

        <div className="mt-4 flex items-center justify-between">
          <div className="font-mono text-xs text-gray-600">
            {isRecording ? (isPaused ? "paused" : "listening") : "mic off"}
          </div>
          <button type="button" className="retro-btn" onClick={toggleRecording}>
            {isRecording ? "Stop" : "Start"}
          </button>
        </div>
      </div>

      <div className="retro-card">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white border border-black rounded-[18px] px-4 py-3 retro-shadow-sm">
            <div className="text-xs font-bold uppercase tracking-wider mb-2">You</div>
            <div className="font-mono text-sm text-gray-800 whitespace-pre-wrap">{finalTranscript || "—"}</div>
          </div>
          <div className="bg-white border border-black rounded-[18px] px-4 py-3 retro-shadow-sm">
            <div className="text-xs font-bold uppercase tracking-wider mb-2">Assistant</div>
            <div className="font-mono text-sm text-gray-800 whitespace-pre-wrap">{assistantText || "—"}</div>
          </div>
        </div>
      </div>
    </div>
  );
};
