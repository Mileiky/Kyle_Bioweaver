import asyncio
from types import SimpleNamespace
from typing import Optional

from taskweaver.app.session_lifecycle import SessionLifecycle


class FakeApp:
    def __init__(self) -> None:
        self.get_calls = []
        self.stop_calls = []
        self.generation = 0

    def get_session(self, session_id: Optional[str] = None):
        self.generation += 1
        resolved_id = session_id or f"new-session-{self.generation}"
        self.get_calls.append(resolved_id)
        return SimpleNamespace(session_id=resolved_id, generation=self.generation)

    def stop_session(self, session_id: str) -> None:
        self.stop_calls.append(session_id)


def test_closed_session_is_reconstructed_with_the_same_id():
    async def scenario():
        app = FakeApp()
        lifecycle = SessionLifecycle(app)

        first = await lifecycle.activate("thread-1", "session-1")
        assert first.session_id == "session-1"

        assert await lifecycle.close("thread-1") == "session-1"

        async with lifecycle.lease("thread-1", "session-1") as restored:
            assert restored.session_id == "session-1"
            assert restored is not first

        assert app.get_calls == ["session-1", "session-1"]
        assert app.stop_calls == ["session-1"]

    asyncio.run(scenario())


def test_close_waits_for_an_in_flight_message_lease():
    async def scenario():
        app = FakeApp()
        lifecycle = SessionLifecycle(app)
        await lifecycle.activate("thread-1", "session-1")

        message_started = asyncio.Event()
        release_message = asyncio.Event()

        async def run_message():
            async with lifecycle.lease("thread-1", "session-1"):
                message_started.set()
                await release_message.wait()

        message_task = asyncio.create_task(run_message())
        await message_started.wait()

        close_task = asyncio.create_task(lifecycle.close("thread-1"))
        await asyncio.sleep(0)
        assert not close_task.done()
        assert app.stop_calls == []

        release_message.set()
        await message_task
        assert await close_task == "session-1"
        assert app.stop_calls == ["session-1"]

    asyncio.run(scenario())


def test_changed_binding_replaces_the_active_session():
    async def scenario():
        app = FakeApp()
        lifecycle = SessionLifecycle(app)

        first = await lifecycle.activate("thread-1", "session-1")
        second = await lifecycle.activate("thread-1", "session-2")

        assert first.session_id == "session-1"
        assert second.session_id == "session-2"
        assert app.stop_calls == ["session-1"]

    asyncio.run(scenario())
