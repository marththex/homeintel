import { ComposeIcon } from "./icons";
import { ConversationList } from "./ConversationList";
import type { ConversationMeta } from "../storage";

interface Props {
  items: ConversationMeta[];
  activeId: string;
  collapsed: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  children?: React.ReactNode;
}

// Desktop-only (hidden below lg via CSS); on mobile the same list opens in a
// bottom sheet. Collapse/expand lives in the Header.
export function Sidebar({ items, activeId, collapsed, onSelect, onNew, onDelete, children }: Props) {
  return (
    <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
      <div className="sidebar-inner">
        <div className="sidebar-top">
          <span className="sidebar-label">Chats</span>
          <button className="icon-btn" onClick={onNew} title="New chat" aria-label="New chat">
            <ComposeIcon />
          </button>
        </div>

        <ConversationList items={items} activeId={activeId} onSelect={onSelect} onDelete={onDelete} />

        {children && <div className="sidebar-filters">{children}</div>}
      </div>
    </aside>
  );
}
