from unittest.mock import MagicMock

from taskweaver.app.app import TaskWeaverApp


def test_stop_session_releases_one_managed_session():
    app = TaskWeaverApp.__new__(TaskWeaverApp)
    app.session_manager = MagicMock()

    app.stop_session("session-1")

    app.session_manager.stop_session.assert_called_once_with("session-1")
