from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional, Protocol

from taskweaver.session.session import Session


class SessionApp(Protocol):
    def get_session(self, session_id: Optional[str] = None) -> Session:
        ...

    def stop_session(self, session_id: str) -> None:
        ...


class SessionLifecycle:
    """Coordinate UI thread lifetimes with reconstructable TaskWeaver sessions."""

    def __init__(self, app: SessionApp) -> None:
        self.app = app
        self.active_sessions: Dict[str, Session] = {}
        self._thread_locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, thread_id: str) -> asyncio.Lock:
        lock = self._thread_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._thread_locks[thread_id] = lock
        return lock

    def _get_or_restore(self, thread_id: str, session_id: Optional[str]) -> Session:
        current = self.active_sessions.get(thread_id)
        if current is not None and (session_id is None or current.session_id == session_id):
            return current

        if current is not None:
            self.app.stop_session(current.session_id)

        restored = self.app.get_session(session_id=session_id)
        self.active_sessions[thread_id] = restored
        return restored

    async def activate(self, thread_id: str, session_id: Optional[str] = None) -> Session:
        """Return an active session, reconstructing it from its persisted ID if needed."""
        async with self._lock_for(thread_id):
            return self._get_or_restore(thread_id, session_id)

    @asynccontextmanager
    async def lease(
        self,
        thread_id: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[Session]:
        """Keep a session alive for the duration of a message or other operation."""
        async with self._lock_for(thread_id):
            yield self._get_or_restore(thread_id, session_id)

    async def close(self, thread_id: str) -> Optional[str]:
        """Release live resources while leaving persisted session data untouched."""
        async with self._lock_for(thread_id):
            session = self.active_sessions.pop(thread_id, None)
            if session is None:
                return None
            self.app.stop_session(session.session_id)
            return session.session_id

    async def delete(self, thread_id: str) -> Optional[str]:
        """Release a thread and forget its lifecycle lock after deletion."""
        session_id = await self.close(thread_id)
        self._thread_locks.pop(thread_id, None)
        return session_id
