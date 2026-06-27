import { useEffect, useState } from "react";
import { getHealth, getStats } from "../api";
import type { HealthResponse, StatsResponse } from "../types";

export function StatusBar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 600);

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 600);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  async function refresh() {
    try {
      const [h, s] = await Promise.all([getHealth(), getStats()]);
      setHealth(h);
      setStats(s);
    } catch {
      setHealth({ status: "degraded", ollama: false, qdrant: false });
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, []);

  const dot = (ok: boolean) => (
    <span
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: ok ? "#4ade80" : "#f87171",
        marginRight: 4,
      }}
    />
  );

  return (
    <div className="status-bar">
      <div className="status-services">
        <span>{dot(health?.ollama ?? false)}{isMobile ? "" : "Ollama"}</span>
        <span>{dot(health?.qdrant ?? false)}{isMobile ? "" : "Qdrant"}</span>
      </div>
      {stats && (
        <div className="status-chunks">
          <span title="Documents">📄{stats.document}</span>
          <span title="Images">🖼️{stats.image}</span>
          <span title="Audio">🎵{stats.audio}</span>
          {!isMobile && <span className="status-total">{stats.total} chunks</span>}
        </div>
      )}
    </div>
  );
}
