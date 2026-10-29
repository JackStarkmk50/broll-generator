"""
SQLite database for projects, chat history, resources, and agent memory.
All data stored locally in data/edit_helper.db.
"""
import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "edit_helper.db"


@dataclass
class Project:
    id: str
    name: str
    description: str = ""
    custom_instructions: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Message:
    id: int
    project_id: str
    role: str           # user | assistant | system
    content: str
    metadata: str = ""  # JSON string
    created_at: str = ""


@dataclass
class Resource:
    id: int
    project_id: str
    resource_type: str  # b-roll | reference | audio | pinterest
    file_path: str = ""
    source_url: str = ""
    title: str = ""
    metadata: str = ""  # JSON string
    created_at: str = ""


@dataclass
class AgentMemoryEntry:
    id: int
    project_id: str
    key: str
    value: str
    created_at: str = ""


class Database:
    """Thread-safe SQLite database manager."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                custom_instructions TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                file_path TEXT DEFAULT '',
                source_url TEXT DEFAULT '',
                title TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS global_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_project ON messages(project_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_resources_project ON resources(project_id);
            CREATE INDEX IF NOT EXISTS idx_memory_project ON agent_memory(project_id, key);
        """)
        conn.commit()
        conn.close()

    # ── Projects ──────────────────────────────────────────────

    def create_project(self, name: str, description: str = "", custom_instructions: str = "") -> Project:
        now = datetime.now(timezone.utc).isoformat()
        project_id = uuid.uuid4().hex[:12]
        conn = self._connect()
        conn.execute(
            "INSERT INTO projects (id, name, description, custom_instructions, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (project_id, name, description, custom_instructions, now, now),
        )
        conn.commit()
        conn.close()
        return Project(id=project_id, name=name, description=description,
                       custom_instructions=custom_instructions, created_at=now, updated_at=now)

    def list_projects(self) -> List[Project]:
        conn = self._connect()
        rows = conn.execute("SELECT * FROM projects ORDER BY updated_at DESC").fetchall()
        conn.close()
        return [Project(**dict(r)) for r in rows]

    def get_project(self, project_id: str) -> Optional[Project]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        conn.close()
        return Project(**dict(row)) if row else None

    def update_project(self, project_id: str, **kwargs) -> bool:
        allowed = {"name", "description", "custom_instructions"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [project_id]
        conn = self._connect()
        conn.execute(f"UPDATE projects SET {set_clause} WHERE id=?", values)
        conn.commit()
        conn.close()
        return True

    def delete_project(self, project_id: str) -> bool:
        conn = self._connect()
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
        conn.commit()
        conn.close()
        return True

    # ── Messages (Chat History) ───────────────────────────────

    def add_message(self, project_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> Message:
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata or {})
        conn = self._connect()
        cur = conn.execute(
            "INSERT INTO messages (project_id, role, content, metadata, created_at) VALUES (?,?,?,?,?)",
            (project_id, role, content, meta_str, now),
        )
        msg_id = cur.lastrowid
        conn.commit()
        conn.close()
        return Message(id=msg_id, project_id=project_id, role=role, content=content,
                       metadata=meta_str, created_at=now)

    def get_messages(self, project_id: str, limit: int = 100) -> List[Message]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM messages WHERE project_id=? ORDER BY created_at ASC LIMIT ?",
            (project_id, limit),
        ).fetchall()
        conn.close()
        return [Message(**dict(r)) for r in rows]

    def clear_messages(self, project_id: str):
        conn = self._connect()
        conn.execute("DELETE FROM messages WHERE project_id=?", (project_id,))
        conn.commit()
        conn.close()

    # ── Resources ─────────────────────────────────────────────

    def add_resource(self, project_id: str, resource_type: str, title: str = "",
                     file_path: str = "", source_url: str = "", metadata: Optional[Dict] = None) -> Resource:
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata or {})
        conn = self._connect()
        cur = conn.execute(
            "INSERT INTO resources (project_id, resource_type, file_path, source_url, title, metadata, created_at) VALUES (?,?,?,?,?,?,?)",
            (project_id, resource_type, file_path, source_url, title, meta_str, now),
        )
        res_id = cur.lastrowid
        conn.commit()
        conn.close()
        return Resource(id=res_id, project_id=project_id, resource_type=resource_type,
                        file_path=file_path, source_url=source_url, title=title,
                        metadata=meta_str, created_at=now)

    def get_resources(self, project_id: str, resource_type: Optional[str] = None) -> List[Resource]:
        conn = self._connect()
        if resource_type:
            rows = conn.execute(
                "SELECT * FROM resources WHERE project_id=? AND resource_type=? ORDER BY created_at DESC",
                (project_id, resource_type),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM resources WHERE project_id=? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        conn.close()
        return [Resource(**dict(r)) for r in rows]

    def delete_resource(self, resource_id: int):
        conn = self._connect()
        conn.execute("DELETE FROM resources WHERE id=?", (resource_id,))
        conn.commit()
        conn.close()

    def get_saved_urls(self, project_id: str) -> set:
        """Return all source_url values already saved in this project (for dedup)."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT source_url FROM resources WHERE project_id=? AND source_url != ''",
            (project_id,),
        ).fetchall()
        conn.close()
        return {r["source_url"] for r in rows}

    def get_saved_download_urls(self, project_id: str) -> set:
        """Return all download_url values from resource metadata (for dedup)."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT metadata FROM resources WHERE project_id=?",
            (project_id,),
        ).fetchall()
        conn.close()
        urls = set()
        for r in rows:
            try:
                parsed = json.loads(r["metadata"]) if r["metadata"] else {}
                if isinstance(parsed, dict):
                    dl = parsed.get("download_url", "")
                    if dl:
                        urls.add(dl)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        return urls

    # ── Agent Memory ──────────────────────────────────────────

    def set_memory(self, project_id: str, key: str, value: str):
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        # Upsert: delete old entry with same key, insert new
        conn.execute("DELETE FROM agent_memory WHERE project_id=? AND key=?", (project_id, key))
        conn.execute(
            "INSERT INTO agent_memory (project_id, key, value, created_at) VALUES (?,?,?,?)",
            (project_id, key, value, now),
        )
        conn.commit()
        conn.close()

    def get_memory(self, project_id: str, key: str) -> Optional[str]:
        conn = self._connect()
        row = conn.execute(
            "SELECT value FROM agent_memory WHERE project_id=? AND key=?",
            (project_id, key),
        ).fetchone()
        conn.close()
        return row["value"] if row else None

    def get_all_memory(self, project_id: str) -> Dict[str, str]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT key, value FROM agent_memory WHERE project_id=? ORDER BY created_at",
            (project_id,),
        ).fetchall()
        conn.close()
        return {r["key"]: r["value"] for r in rows}

    def delete_memory(self, project_id: str, key: str):
        conn = self._connect()
        conn.execute("DELETE FROM agent_memory WHERE project_id=? AND key=?", (project_id, key))
        conn.commit()
        conn.close()

    # ── Global Settings ───────────────────────────────────────

    def get_setting(self, key: str, default: str = "") -> str:
        conn = self._connect()
        row = conn.execute("SELECT value FROM global_settings WHERE key=?", (key,)).fetchone()
        conn.close()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO global_settings (key, value) VALUES (?,?)",
            (key, value),
        )
        conn.commit()
        conn.close()

    def get_all_settings(self) -> Dict[str, str]:
        conn = self._connect()
        rows = conn.execute("SELECT key, value FROM global_settings").fetchall()
        conn.close()
        return {r["key"]: r["value"] for r in rows}


# Singleton
_db_instance: Optional[Database] = None

def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
