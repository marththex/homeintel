import { useCallback, useRef, useState } from "react";

export type RecorderState = "idle" | "recording" | "error";

export interface UseRecorder {
  state: RecorderState;
  seconds: number;
  error: string | null;
  start: () => Promise<void>;
  stop: () => Promise<Blob | null>;
  cancel: () => void;
}

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/aac"];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t));
}

export function useRecorder(): UseRecorder {
  const [state, setState] = useState<RecorderState>("idle");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);
  const resolveRef = useRef<((b: Blob | null) => void) | null>(null);

  const cleanup = useCallback(() => {
    if (timerRef.current !== null) { clearInterval(timerRef.current); timerRef.current = null; }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    recorderRef.current = null;
    chunksRef.current = [];
  }, []);

  const start = useCallback(async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setState("error");
      setError("Microphone needs HTTPS — see Tailscale setup.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = pickMimeType();
      const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      cancelledRef.current = false;
      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = () => {
        if (timerRef.current !== null) { clearInterval(timerRef.current); timerRef.current = null; }
        const blob = cancelledRef.current
          ? null
          : new Blob(chunksRef.current, { type: rec.mimeType || "audio/webm" });
        cleanup();
        setState("idle");
        resolveRef.current?.(blob);
        resolveRef.current = null;
      };
      rec.onerror = () => {
        if (timerRef.current !== null) { clearInterval(timerRef.current); timerRef.current = null; }
        cleanup();
        setState("error");
        setError("Recording failed.");
        resolveRef.current?.(null);
        resolveRef.current = null;
      };
      recorderRef.current = rec;
      rec.start();
      setSeconds(0);
      setState("recording");
      timerRef.current = window.setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch (e) {
      cleanup();
      setState("error");
      setError(
        (e as Error).name === "NotAllowedError"
          ? "Microphone permission denied."
          : "Could not start recording."
      );
    }
  }, [cleanup]);

  const stop = useCallback(() => {
    return new Promise<Blob | null>((resolve) => {
      const rec = recorderRef.current;
      if (!rec || rec.state === "inactive") { resolve(null); return; }
      cancelledRef.current = false;
      resolveRef.current = resolve;
      rec.stop();
    });
  }, []);

  const cancel = useCallback(() => {
    const rec = recorderRef.current;
    cancelledRef.current = true;
    if (rec && rec.state !== "inactive") {
      rec.stop();
    } else {
      cleanup();
      setState("idle");
    }
  }, [cleanup]);

  return { state, seconds, error, start, stop, cancel };
}
