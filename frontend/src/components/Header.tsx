import { ComposeIcon } from "./icons";
import type { HealthResponse } from "../types";

interface Props {
  health: HealthResponse | null;
  onStatusClick: () => void;
  onNewChat: () => void;
}

export function Header({ health, onStatusClick, onNewChat }: Props) {
  const healthy = health?.status === "ok";

  return (
    <header className="header">
      <h1 className="logo">HomeIntel</h1>
      <div className="header-actions">
        <button
          className="health-pill"
          onClick={onStatusClick}
          title="System status"
          aria-label="System status"
        >
          <span className={`health-dot ${healthy ? "ok" : "bad"}`} />
        </button>
        <button
          className="icon-btn"
          onClick={onNewChat}
          title="New chat"
          aria-label="New chat"
        >
          <ComposeIcon />
        </button>
      </div>
    </header>
  );
}
