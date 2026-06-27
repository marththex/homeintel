import { useEffect, useState } from "react";
import { getHealth, getStats } from "../api";
import type { HealthResponse, StatsResponse } from "../types";

export function useSystemStatus() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);

  async function refresh() {
    try {
      const [h, s] = await Promise.all([getHealth(), getStats()]);
      setHealth(h);
      setStats(s);
    } catch {
      setHealth({ status: "degraded", ollama: false, qdrant: false });
      setStats(null);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, []);

  return { health, stats, refresh };
}
