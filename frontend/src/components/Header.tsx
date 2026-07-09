import { ComposeIcon, SidebarIcon } from "./icons";
import type { HealthResponse } from "../types";

interface Props {
  health: HealthResponse | null;
  onStatusClick: () => void;
  onNewChat: () => void;
  onToggleSidebar?: () => void;
}

export function Header({ health, onStatusClick, onNewChat, onToggleSidebar }: Props) {
  const healthy = health?.status === "ok";

  return (
    <header className="header">
      <div className="logo-group">
        {onToggleSidebar && (
          <button
            className="icon-btn"
            onClick={onToggleSidebar}
            title="Chats"
            aria-label="Chats"
          >
            <SidebarIcon />
          </button>
        )}
        <img src="/icon-512.png" alt="" className="logo-mark" />
        <h1 className="logo">HomeIntel</h1>
      </div>
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
