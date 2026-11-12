"""Event-sourced sessions, persistence, replay, and context projection."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol
from uuid import uuid4


SESSION_SCHEMA_VERSION = 1
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionMigrationRegistry:
    """Ordered migrations for persisted Session payloads and events."""

    def __init__(self):
        self._session_migrations: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self._event_migrations: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def register_session(self, from_version: int, migration: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._session_migrations[from_version] = migration

    def register_event(self, from_version: int, migration: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._event_migrations[from_version] = migration

    def migrate(self, value: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(value)
        version = int(payload.get("schema_version", 1))
        if version > SESSION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Session schema_version={version}; maximum supported={SESSION_SCHEMA_VERSION}"
            )
        while version < SESSION_SCHEMA_VERSION:
            migration = self._session_migrations.get(version)
            if migration is None:
                raise ValueError(f"missing Session migration from schema_version={version}")
            payload = migration(payload)
            version = int(payload.get("schema_version", version + 1))
        payload["events"] = [self.migrate_event(event) for event in payload.get("events", [])]
        return payload

    def migrate_event(self, value: dict[str, Any]) -> dict[str, Any]:
        payload = deepcopy(value)
        version = int(payload.get("schema_version", 1))
        if version > SESSION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported SessionEvent schema_version={version}; maximum supported={SESSION_SCHEMA_VERSION}"
            )
        while version < SESSION_SCHEMA_VERSION:
            migration = self._event_migrations.get(version)
            if migration is None:
                raise ValueError(f"missing SessionEvent migration from schema_version={version}")
            payload = migration(payload)
            version = int(payload.get("schema_version", version + 1))
        return payload


session_migrations = SessionMigrationRegistry()


def _json_safe(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-safe copy while removing common credential fields."""
    if key is not None and key.lower() in _SENSITIVE_KEYS:
        return "[redacted]"                                    # 敏感字段直接脱敏
    if value is None or isinstance(value, (str, int, float, bool)):
        return value                                            # 基础类型原样返回
    if is_dataclass(value):
        return _json_safe(asdict(value))                        # dataclass 转字典再递归
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict())                  # 有 to_dict 的调它
        except TypeError:
            pass
    if isinstance(value, dict):
        return {str(item_key): _json_safe(item, key=str(item_key)) for item_key, item in value.items()}   # 字典逐项递归
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]              # 序列逐项递归
    return repr(value)                                           # 兜底：转成字符串表示
    # 例如输入：
    # s.append("test.event", {
    # "api_key": "sk-real-secret",
    # "Authorization": "Bearer abc",
    # "city": "南京",
    # "nested": {"secret": "inner", "ok": 1},
    # })
    # 处理后：
    # {
    #     "api_key": "[redacted]",
    #     "Authorization": "[redacted]",
    #     "city": "南京",
    #     "nested": {"secret": "[redacted]", "ok": 1}
    # }

@dataclass
class SessionEvent:
    """One immutable fact in a session event log."""

    type: str                                                       # 发生了什么
    session_id: str                                                 # 属于哪个会话
    data: dict[str, Any] = field(default_factory=dict)              # 具体内容
    event_id: str = field(default_factory=lambda: uuid4().hex)      # 事件id          定位
    timestamp: str = field(default_factory=_utc_now)                # 什么时间         定位
    sequence: int = 0                                               # 第几条           定位
    schema_version: int = SESSION_SCHEMA_VERSION
    turn_id: str | None = None                                      # 哪一轮
    step_id: str | None = None                                      # 哪个步骤
    run_id: str | None = None                                       # 哪次run
    agent_id: str | None = None                                     # 哪个agent
    parent_event_id: str | None = None                              # 由哪条事件引发的    因果

    def __post_init__(self) -> None:
        if not self.type or not isinstance(self.type, str):   # 必须是不为空的string类型
            raise ValueError("SessionEvent.type must be a non-empty string")
        if not self.session_id:
            raise ValueError("SessionEvent.session_id must not be empty")
        if self.sequence < 0:
            raise ValueError("SessionEvent.sequence must be non-negative")
        if self.schema_version < 1:
            raise ValueError("SessionEvent.schema_version must be at least 1")
        self.data = _json_safe(self.data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "type": self.type,
            "data": deepcopy(self.data),
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "schema_version": self.schema_version,
            "turn_id": self.turn_id,
            "step_id": self.step_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "parent_event_id": self.parent_event_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SessionEvent":
        if not isinstance(value, dict):
            raise TypeError("SessionEvent payload must be a dictionary")
        return cls(
            event_id=str(value.get("event_id") or uuid4().hex),
            session_id=str(value["session_id"]),
            type=str(value["type"]),
            data=dict(value.get("data") or {}),
            timestamp=str(value.get("timestamp") or _utc_now()),
            sequence=int(value.get("sequence", 0)),
            schema_version=int(value.get("schema_version", SESSION_SCHEMA_VERSION)),
            turn_id=value.get("turn_id"),
            step_id=value.get("step_id"),
            run_id=value.get("run_id"),
            agent_id=value.get("agent_id"),
            parent_event_id=value.get("parent_event_id"),
        )


@dataclass
class SessionCheckpoint:
    checkpoint_id: str
    session_id: str
    sequence: int
    label: str | None = None
    created_at: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class SessionReplay:
    session_id: str
    event_count: int
    completed_turns: list[str] = field(default_factory=list)
    incomplete_turns: list[str] = field(default_factory=list)
    failed_turns: list[str] = field(default_factory=list)
    last_sequence: int = 0
    last_event_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Session:
    """Append-only session state reconstructed from versioned events."""

    session_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[SessionEvent] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    schema_version: int = SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("Session.session_id must not be empty")
        self.metadata = _json_safe(self.metadata)
        self.events = sorted(list(self.events), key=lambda event: event.sequence)
        self.validate()

    @property
    def next_sequence(self) -> int:
        return self.events[-1].sequence + 1 if self.events else 1

    def append(
            self,
            event_type: str,
            data: dict[str, Any] | None = None,
            *,
            turn_id: str | None = None,
            step_id: str | None = None,
            run_id: str | None = None,
            agent_id: str | None = None,
            parent_event_id: str | None = None,
    ) -> SessionEvent:
        event = SessionEvent(
            type=event_type,
            session_id=self.session_id,
            data=data or {},
            sequence=self.next_sequence,        # 自动计算序号
            schema_version=self.schema_version,
            turn_id=turn_id,                    # 从self._current_turn_id传进来
            step_id=step_id,                    # 从self._current_run_id传进来
            run_id=run_id,
            agent_id=agent_id,
            parent_event_id=parent_event_id,
        )
        self.events.append(event)               # 进列表
        self.updated_at = event.timestamp       # 更新会话时间
        return event

    # 接收外部已经构造好的事件，验证后放进会话。为持久化存储（SQLite、文件、远端）预留的接口
    def append_event(self, event: SessionEvent) -> SessionEvent:
        if event.session_id != self.session_id:
            raise ValueError("event session_id does not match Session")
        expected = self.next_sequence
        if event.sequence == 0:
            event.sequence = expected
        if event.sequence != expected:
            raise ValueError(f"event sequence must be {expected}, got {event.sequence}")
        if any(existing.event_id == event.event_id for existing in self.events):
            raise ValueError(f"duplicate event_id: {event.event_id}")
        self.events.append(event)
        self.updated_at = event.timestamp
        return event

    def page(self, *, after: int = 0, limit: int = 100) -> list[SessionEvent]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        return [event for event in self.events if event.sequence > after][:limit]

    ''' checkpoint 标记处理到第几条了'''
    def checkpoint(self, label: str | None = None, metadata: dict[str, Any] | None = None) -> SessionCheckpoint:
        checkpoint = SessionCheckpoint(
            checkpoint_id=uuid4().hex,
            session_id=self.session_id,
            sequence=self.events[-1].sequence if self.events else 0,
            label=label,
            metadata=_json_safe(metadata or {}),
        )
        self.append("session.checkpointed", {"checkpoint": checkpoint.to_dict()})
        return checkpoint

    """复制一份历史，然后各走各的"""
    def fork(
            self,
            *,
            new_session_id: str | None = None,
            through_sequence: int | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> "Session":
        boundary = through_sequence if through_sequence is not None else (self.events[-1].sequence if self.events else 0)
        if boundary < 0 or boundary > (self.events[-1].sequence if self.events else 0):
            raise ValueError("through_sequence is outside the session event range")
        forked = Session(
            session_id=new_session_id or uuid4().hex,
            metadata={
                **deepcopy(self.metadata),
                **(metadata or {}),
                "forked_from": self.session_id,
                "forked_at_sequence": boundary,
            },
        )
        forked.append("session.started", {"forked_from": self.session_id, "through_sequence": boundary})
        for event in self.events:
            if event.sequence > boundary:
                break
            # 新会话的sequence重新从1排
            forked.append(
                event.type,
                {
                    **deepcopy(event.data),                # 深拷贝内容
                    "source_session_id": self.session_id,  # 记来源会话
                    "source_event_id": event.event_id,     # 能追回原会话
                    "source_sequence": event.sequence,     # 记来源序号
                },
                turn_id=event.turn_id,                     # 归属字段原样复制
                step_id=event.step_id,
                run_id=event.run_id,
                agent_id=event.agent_id,
                parent_event_id=event.event_id,            # 指向原事件
            )
        # 用append而不是events.append是因为前者自动分配新sequence和event_id后者会继续沿用原sequence，validate会报错
        forked.append("session.forked", {"source_session_id": self.session_id, "through_sequence": boundary})
        return forked

    """用来判断轮次状态的"""
    def replay(self) -> SessionReplay:
        started: set[str] = set()
        completed: set[str] = set()
        failed: set[str] = set()
        for event in self.events:
            if not event.turn_id:
                continue
            if event.type == "turn.started":
                started.add(event.turn_id)
            elif event.type == "turn.completed":
                completed.add(event.turn_id)
            elif event.type == "turn.failed":
                failed.add(event.turn_id)
        return SessionReplay(
            session_id=self.session_id,
            event_count=len(self.events),
            completed_turns=sorted(completed),
            incomplete_turns=sorted(started - completed - failed),  # 开始了但既没完成也没失败
            failed_turns=sorted(failed),
            last_sequence=self.events[-1].sequence if self.events else 0,
            last_event_type=self.events[-1].type if self.events else None,
        )

    """
    负责检验事件流是否可信。
    三个地方被调用：构造session时， fork之后， 手动调用
    """
    def validate(self) -> None:
        seen_ids: set[str] = set()
        expected = 1
        for event in self.events:
            if event.session_id != self.session_id:  # 事件属于别的会话
                raise ValueError("session contains an event for another session")
            if event.event_id in seen_ids:           # 事件id重复
                raise ValueError(f"duplicate event_id: {event.event_id}")
            if event.sequence != expected:           # 序号断号
                raise ValueError(f"event sequence gap: expected {expected}, got {event.sequence}")
            if event.schema_version > SESSION_SCHEMA_VERSION:   # 版本比代码新
                raise ValueError(
                    f"unsupported SessionEvent schema_version={event.schema_version}; "
                    f"maximum supported={SESSION_SCHEMA_VERSION}"
                )
            seen_ids.add(event.event_id)
            expected += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "metadata": deepcopy(self.metadata),
            "events": [event.to_dict() for event in self.events],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Session":
        value = session_migrations.migrate(value)
        return cls(
            session_id=str(value["session_id"]),
            metadata=dict(value.get("metadata") or {}),
            events=[SessionEvent.from_dict(event) for event in value.get("events", [])],
            created_at=str(value.get("created_at") or _utc_now()),
            updated_at=str(value.get("updated_at") or _utc_now()),
            schema_version=int(value.get("schema_version", SESSION_SCHEMA_VERSION)),
        )


class SessionStore(Protocol):
    def create(self, session: Session | None = None, *, metadata: dict[str, Any] | None = None) -> Session:
        ...

    def get(self, session_id: str) -> Session | None:
        ...

    def save(self, session: Session) -> None:
        ...

    def list(self, *, limit: int = 100) -> list[Session]:
        ...

    def delete(self, session_id: str) -> bool:
        ...


class InMemorySessionStore:
    """Thread-safe dependency-free SessionStore."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def create(self, session: Session | None = None, *, metadata: dict[str, Any] | None = None) -> Session:
        with self._lock:
            value = session or Session(metadata=metadata or {})
            if value.session_id in self._sessions:
                raise ValueError(f"session already exists: {value.session_id}")
            self._sessions[value.session_id] = deepcopy(value)
            return deepcopy(value)

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            value = self._sessions.get(str(session_id))
            return deepcopy(value) if value is not None else None

    def save(self, session: Session) -> None:
        session.validate()
        with self._lock:
            current = self._sessions.get(session.session_id)
            if current and current.events and session.events:
                if session.events[-1].sequence < current.events[-1].sequence:
                    raise ValueError("refusing to replace a Session with an older event sequence")
            self._sessions[session.session_id] = deepcopy(session)

    def list(self, *, limit: int = 100) -> list[Session]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._lock:
            values = sorted(self._sessions.values(), key=lambda item: item.updated_at, reverse=True)
            return deepcopy(values[:limit])

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(str(session_id), None) is not None


class JsonlSessionStore:
    """One append-readable JSONL file per Session with atomic full saves."""

    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _safe_id(session_id: str) -> str:
        value = str(session_id)
        if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value):
            raise ValueError("session_id contains unsafe characters")
        return value

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{self._safe_id(session_id)}.jsonl"

    def create(self, session: Session | None = None, *, metadata: dict[str, Any] | None = None) -> Session:
        value = session or Session(metadata=metadata or {})
        with self._lock:
            if self._path(value.session_id).exists():
                raise ValueError(f"session already exists: {value.session_id}")
            self.save(value)
        return deepcopy(value)

    def get(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        with self._lock, path.open("r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        if not lines or lines[0].get("kind") != "session":
            raise ValueError(f"invalid Session JSONL header: {path}")
        header = lines[0]["value"]
        events = [SessionEvent.from_dict(line["value"]) for line in lines[1:] if line.get("kind") == "event"]
        return Session(
            session_id=header["session_id"],
            metadata=header.get("metadata") or {},
            events=events,
            created_at=header.get("created_at") or _utc_now(),
            updated_at=header.get("updated_at") or _utc_now(),
            schema_version=int(header.get("schema_version", SESSION_SCHEMA_VERSION)),
        )

    def save(self, session: Session) -> None:
        session.validate()
        path = self._path(session.session_id)
        temporary = path.with_suffix(f".tmp-{uuid4().hex}")
        header = {
            "session_id": session.session_id,
            "metadata": session.metadata,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "schema_version": session.schema_version,
        }
        with self._lock:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": "session", "value": header}, ensure_ascii=False) + "\n")
                for event in session.events:
                    handle.write(json.dumps({"kind": "event", "value": event.to_dict()}, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)

    def list(self, *, limit: int = 100) -> list[Session]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        sessions = []
        for path in sorted(self.directory.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True):
            session = self.get(path.stem)
            if session is not None:
                sessions.append(session)
            if len(sessions) >= limit:
                break
        return sessions

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            return True


class SqliteSessionStore:
    """Standard-library SQLite SessionStore with transactional saves."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_events (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )

    def create(self, session: Session | None = None, *, metadata: dict[str, Any] | None = None) -> Session:
        value = session or Session(metadata=metadata or {})
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (value.session_id,)
            ).fetchone()
            if existing:
                raise ValueError(f"session already exists: {value.session_id}")
        self.save(value)
        return deepcopy(value)

    def get(self, session_id: str) -> Session | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (str(session_id),)
            ).fetchone()
            if row is None:
                return None
            event_rows = connection.execute(
                "SELECT payload FROM session_events WHERE session_id = ? ORDER BY sequence",
                (str(session_id),),
            ).fetchall()
        return Session(
            session_id=row["session_id"],
            metadata=json.loads(row["metadata"]),
            events=[SessionEvent.from_dict(json.loads(event_row["payload"])) for event_row in event_rows],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            schema_version=row["schema_version"],
        )

    def save(self, session: Session) -> None:
        session.validate()
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(
                """
                INSERT INTO sessions(session_id, metadata, created_at, updated_at, schema_version)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    metadata=excluded.metadata,
                    updated_at=excluded.updated_at,
                    schema_version=excluded.schema_version
                """,
                (
                    session.session_id,
                    json.dumps(session.metadata, ensure_ascii=False),
                    session.created_at,
                    session.updated_at,
                    session.schema_version,
                ),
            )
            connection.execute("DELETE FROM session_events WHERE session_id = ?", (session.session_id,))
            connection.executemany(
                "INSERT INTO session_events(session_id, sequence, event_id, payload) VALUES (?, ?, ?, ?)",
                [
                    (
                        session.session_id,
                        event.sequence,
                        event.event_id,
                        json.dumps(event.to_dict(), ensure_ascii=False),
                    )
                    for event in session.events
                ],
            )

    def list(self, *, limit: int = 100) -> list[Session]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [session for row in rows if (session := self.get(row["session_id"])) is not None]

    def delete(self, session_id: str) -> bool:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            cursor = connection.execute("DELETE FROM sessions WHERE session_id = ?", (str(session_id),))
            return cursor.rowcount > 0


class ContextProjector:
    """Project model messages and runtime views from Session events."""
    """Session 负责存事件，ContextProjector 负责把这些事件翻译成 messages"""
    def messages(self, session: Session, *, turn_id: str | None = None) -> list[dict[str, Any]]:
        requested = [
            event for event in session.events
            if event.type == "model.requested"
            and (turn_id is None or event.turn_id == turn_id)
            and isinstance(event.data.get("messages"), list)
        ]
        messages: list[dict[str, Any]] = []
        for event in session.events:
            if turn_id is not None and event.turn_id != turn_id:
                continue
            if event.type in {"message.received", "message.created"}:
                messages.append({
                    "role": event.data.get("role", "user"),
                    # 显式 None 也要转成空串：provider 不接受 content=null
                    "content": event.data.get("content") or ""
                })
            elif event.type in {"assistant.completed", "model.completed"} and event.data.get("content") is not None:
                messages.append({
                    "role": "assistant",
                    "content": event.data.get("content", "")
                })
            # 故意跳过 tool.completed：
            # 工具结果必须紧跟一条带 tool_calls 的 assistant 消息才合法，
            # 而事件里没有存 tool_calls，投影出来就是"孤儿 tool 消息"，
            # 实测会让 DeepSeek 直接 422（messages[n]: invalid type: null）。
            # 历史里保留"问-答"文本已经足够让模型接上下文，工具结果不需要回放。
        if messages:
            return messages
        if requested:
            return deepcopy(requested[-1].data["messages"])
        return []

    def model_request(
            self,
            session: Session,
            *,
            sequence: int | None = None,
    ) -> list[dict[str, Any]]:
        """Reconstruct the exact messages sent for one persisted model request."""
        requested = [
            event for event in session.events
            if event.type == "model.requested"
            and isinstance(event.data.get("messages"), list)
            and (sequence is None or event.sequence == sequence)
        ]
        if not requested:
            raise LookupError("model request event was not found")
        return deepcopy(requested[-1].data["messages"])

    def trace(self, session: Session, *, turn_id: str | None = None) -> list[dict[str, Any]]:
        trace = []
        for event in session.events:
            if turn_id is not None and event.turn_id != turn_id:
                continue
            trace_type = event.data.get("trace_type")
            if trace_type:
                trace.append({
                    "type": trace_type,
                    "data": deepcopy(event.data.get("trace_data") or {}),
                    "timestamp": event.timestamp,
                    "trace_id": event.data.get("trace_id"),
                    "parent_trace_id": event.data.get("parent_trace_id"),
                    "run_group_id": event.data.get("run_group_id"),
                })
        return trace


@dataclass
class CompactionResult:
    messages: list[dict[str, Any]]
    removed_count: int
    summary: str | None = None
    spilled: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ContextBudget:
    """Dependency-free context budget using a conservative character estimate."""

    max_tokens: int = 8192
    reserved_output_tokens: int = 1024
    chars_per_token: float = 4.0
    # 工具 schema 每轮都会随请求发送，但不属于 messages。
    # 不把它预留出来，预算判断会系统性偏乐观（实测 19 个工具约 3.5k token）。
    reserved_tools_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens < 1 or self.reserved_output_tokens < 0 or self.reserved_tools_tokens < 0:
            raise ValueError("context token limits must be non-negative")
        if self.reserved_output_tokens + self.reserved_tools_tokens >= self.max_tokens:
            raise ValueError("reserved tokens must be smaller than max_tokens")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")

    def estimate(self, messages: Iterable[dict[str, Any]]) -> int:
        rendered = json.dumps(list(messages), ensure_ascii=False, sort_keys=True)
        return max(1, int(len(rendered) / self.chars_per_token) + 1)

    def available_tokens(self) -> int:
        return self.max_tokens - self.reserved_output_tokens - self.reserved_tools_tokens

    def fits(self, messages: Iterable[dict[str, Any]]) -> bool:
        return self.estimate(messages) <= self.available_tokens()


class ContextCompactor:
    """Deterministic message trimming with optional summary generation."""

    def __init__(
            self,
            summarizer: Callable[[list[dict[str, Any]]], str] | None = None,
            *,
            max_inline_tool_chars: int = 12000,
    ):
        self.summarizer = summarizer
        self.max_inline_tool_chars = max_inline_tool_chars

    def compact(self, messages: Iterable[dict[str, Any]], *, max_messages: int = 20) -> CompactionResult:
        values, spilled = self._spill_tool_results(messages)
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        if len(values) <= max_messages:
            return CompactionResult(messages=values, removed_count=0, spilled=spilled)

        system_messages = [message for message in values if message.get("role") == "system"]
        system = system_messages[:1]
        # 除第一条以外的 system 消息（通常是上一次生成的摘要）也要参与本轮摘要，
        # 否则连续压缩时旧摘要会被静默丢弃，早前信息就真丢了。
        extra_system = system_messages[1:]
        non_system = [message for message in values if message.get("role") != "system"]
        keep_count = max_messages - len(system)
        kept = non_system[-keep_count:]
        removed = extra_system + non_system[:-keep_count]

        while kept and kept[0].get("role") == "tool":
            removed.append(kept.pop(0))
        summary = self.summarizer(deepcopy(removed)) if self.summarizer and removed else None
        if summary:
            summary_message = {
                "role": "system",
                "content": f"Previous conversation summary:\n{summary}",
                "metadata": {"compacted": True, "removed_count": len(removed)},
            }
            result_messages = system + [summary_message] + kept
        else:
            result_messages = system + kept
        return CompactionResult(
            messages=result_messages,
            removed_count=len(removed),
            summary=summary,
            spilled=spilled,
        )

    def compact_to_budget(
            self,
            messages: Iterable[dict[str, Any]],
            budget: ContextBudget,
    ) -> CompactionResult:
        values = list(messages)
        if budget.fits(values):
            prepared, spilled = self._spill_tool_results(values)
            return CompactionResult(messages=prepared, removed_count=0, spilled=spilled)

        # 先用"不生成摘要"的模式探出最少保留多少条，再做一次真正的压缩。
        # 原来的写法每试一个条数就调一次摘要模型：25 条消息要试 14 次，
        # 每次摘要约 1.8 秒，一次压缩就拖了 16 秒——而它恰好发生在上下文最满、
        # 用户等得最久的时候。探路时不算摘要，只需 1 次摘要调用。
        summarizer = self.summarizer
        # 探路时给摘要本身预留 250 token：否则算出来的条数刚好卡满，
        # 加上摘要又超了，最后只能放弃摘要（实测就发生过）。
        probe_budget = ContextBudget(
            max_tokens=budget.max_tokens,
            reserved_output_tokens=budget.reserved_output_tokens + 250,
            chars_per_token=budget.chars_per_token,
        )
        try:
            self.summarizer = None
            probe = max(2, len(values))
            while probe > 2:
                if probe_budget.fits(self.compact(values, max_messages=probe).messages):
                    break
                probe -= 1
        finally:
            self.summarizer = summarizer

        result = self.compact(values, max_messages=probe)
        if budget.fits(result.messages):
            return result

        # 摘要本身要占 token，可能刚好把它顶出预算：退一步再压一次（最多多一次摘要调用）
        if probe > 2:
            tighter = self.compact(values, max_messages=probe - 1)
            if budget.fits(tighter.messages):
                return tighter

        # 还不行就放弃摘要，用纯裁剪保证"一定放得下"
        self.summarizer = None
        try:
            return self.compact(values, max_messages=max(2, probe - 1))
        finally:
            self.summarizer = summarizer

    def _spill_tool_results(
            self,
            messages: Iterable[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prepared: list[dict[str, Any]] = []
        spilled: list[dict[str, Any]] = []
        for message in messages:
            value = deepcopy(message)
            content = value.get("content")
            if value.get("role") == "tool" and isinstance(content, str) and len(content) > self.max_inline_tool_chars:
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                reference = f"tool-result:sha256:{digest}"
                spilled.append({
                    "reference": reference,
                    "sha256": digest,
                    "size": len(content),
                    "content": content,
                })
                value["content"] = f"[Tool result stored outside context: {reference}, {len(content)} chars]"
                value.setdefault("metadata", {})["content_reference"] = reference
            prepared.append(value)
        return prepared, spilled


__all__ = [
    "SESSION_SCHEMA_VERSION",
    "SessionMigrationRegistry",
    "session_migrations",
    "SessionEvent",
    "SessionCheckpoint",
    "SessionReplay",
    "Session",
    "SessionStore",
    "InMemorySessionStore",
    "JsonlSessionStore",
    "SqliteSessionStore",
    "ContextProjector",
    "CompactionResult",
    "ContextBudget",
    "ContextCompactor",
]
