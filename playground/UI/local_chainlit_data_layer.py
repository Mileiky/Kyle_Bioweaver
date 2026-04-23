import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from chainlit.context import context
from chainlit.data import BaseDataLayer, queue_until_user_message
from chainlit.types import PageInfo, PaginatedResponse, Pagination, ThreadDict, ThreadFilter
from chainlit.user import PersistedUser, User


class LocalChainlitDataLayer(BaseDataLayer):
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        self.users_path = os.path.join(base_path, "users")
        os.makedirs(self.users_path, exist_ok=True)

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _read_json(self, path: str, default: Any = None) -> Any:
        if not os.path.exists(path):
            return default
        with open(path, "r") as f:
            return json.load(f)

    def _write_json(self, path: str, value: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(value, f)

    def _user_dir(self, user_id: str) -> str:
        return os.path.join(self.users_path, user_id)

    def _thread_dir(self, user_id: str, thread_id: str) -> str:
        return os.path.join(self._user_dir(user_id), "threads", thread_id)

    def _thread_path(self, user_id: str, thread_id: str) -> str:
        return os.path.join(self._thread_dir(user_id, thread_id), "thread.json")

    def _steps_dir(self, user_id: str, thread_id: str) -> str:
        return os.path.join(self._thread_dir(user_id, thread_id), "steps")

    def _elements_dir(self, user_id: str, thread_id: str) -> str:
        return os.path.join(self._thread_dir(user_id, thread_id), "elements")

    def _feedback_dir(self, user_id: str, thread_id: str) -> str:
        return os.path.join(self._thread_dir(user_id, thread_id), "feedback")

    def _identity_path(self, user_id: str) -> str:
        return os.path.join(self._user_dir(user_id), "identity.json")

    def _find_thread_owner(self, thread_id: str) -> Optional[str]:
        for user_id in os.listdir(self.users_path):
            candidate = self._thread_path(user_id, thread_id)
            if os.path.exists(candidate):
                return user_id
        return None

    def _load_thread(self, user_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
        return self._read_json(self._thread_path(user_id, thread_id))

    def _load_steps(self, user_id: str, thread_id: str) -> List[Dict[str, Any]]:
        steps_dir = self._steps_dir(user_id, thread_id)
        if not os.path.exists(steps_dir):
            return []
        steps: List[Dict[str, Any]] = []
        for name in os.listdir(steps_dir):
            if not name.endswith(".json"):
                continue
            item = self._read_json(os.path.join(steps_dir, name))
            if item is not None:
                steps.append(item)
        steps.sort(key=lambda s: s.get("createdAt") or s.get("start") or "")
        return steps

    def _load_elements(self, user_id: str, thread_id: str) -> List[Dict[str, Any]]:
        elements_dir = self._elements_dir(user_id, thread_id)
        if not os.path.exists(elements_dir):
            return []
        elements: List[Dict[str, Any]] = []
        for name in os.listdir(elements_dir):
            if not name.endswith(".json"):
                continue
            item = self._read_json(os.path.join(elements_dir, name))
            if item is not None:
                elements.append(item)
        return elements

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        user_data = self._read_json(self._identity_path(identifier))
        if user_data is None:
            return None
        return PersistedUser(
            id=user_data["id"],
            identifier=user_data["identifier"],
            createdAt=user_data["createdAt"],
            metadata=user_data.get("metadata", {}),
        )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        if user.identifier is None:
            raise ValueError("Creation of anonymous users is not supported")
        ts = self._now()
        item = {
            "id": user.identifier,
            "identifier": user.identifier,
            "metadata": user.metadata or {},
            "createdAt": ts,
        }
        self._write_json(self._identity_path(str(user.identifier)), item)
        return PersistedUser(
            id=item["id"],
            identifier=item["identifier"],
            createdAt=item["createdAt"],
            metadata=item["metadata"],
        )

    async def upsert_feedback(self, feedback: Any) -> str:
        feedback_dict = dict(feedback)
        thread_id = feedback_dict["threadId"]
        user_id = self._find_thread_owner(thread_id) or context.session.user.identifier
        feedback_id = feedback_dict["id"]
        path = os.path.join(self._feedback_dir(user_id, thread_id), f"{feedback_id}.json")
        self._write_json(path, feedback_dict)
        return feedback_id

    async def delete_feedback(self, feedback_id: str) -> bool:
        for user_id in os.listdir(self.users_path):
            threads_root = os.path.join(self._user_dir(user_id), "threads")
            if not os.path.exists(threads_root):
                continue
            for thread_id in os.listdir(threads_root):
                path = os.path.join(self._feedback_dir(user_id, thread_id), f"{feedback_id}.json")
                if os.path.exists(path):
                    os.remove(path)
                    return True
        return False

    @queue_until_user_message()
    async def create_element(self, element_dict: Dict[str, Any]):
        thread_id = element_dict["threadId"]
        user_id = self._find_thread_owner(thread_id) or context.session.user.identifier
        element_id = element_dict["id"]
        path = os.path.join(self._elements_dir(user_id, thread_id), f"{element_id}.json")
        self._write_json(path, dict(element_dict))

    async def get_element(self, thread_id: str, element_id: str) -> Optional[Dict[str, Any]]:
        user_id = self._find_thread_owner(thread_id)
        if user_id is None:
            return None
        path = os.path.join(self._elements_dir(user_id, thread_id), f"{element_id}.json")
        return self._read_json(path)

    async def delete_element(self, element_id: str):
        for user_id in os.listdir(self.users_path):
            threads_root = os.path.join(self._user_dir(user_id), "threads")
            if not os.path.exists(threads_root):
                continue
            for thread_id in os.listdir(threads_root):
                path = os.path.join(self._elements_dir(user_id, thread_id), f"{element_id}.json")
                if os.path.exists(path):
                    os.remove(path)
                    return

    @queue_until_user_message()
    async def create_step(self, step_dict: Dict[str, Any]):
        thread_id = step_dict["threadId"]
        user_id = self._find_thread_owner(thread_id) or context.session.user.identifier
        step_id = step_dict["id"]
        path = os.path.join(self._steps_dir(user_id, thread_id), f"{step_id}.json")
        self._write_json(path, dict(step_dict))

    @queue_until_user_message()
    async def update_step(self, step_dict: Dict[str, Any]):
        await self.create_step(step_dict)

    @queue_until_user_message()
    async def delete_step(self, step_id: str):
        thread_id = context.session.thread_id
        user_id = self._find_thread_owner(thread_id) or context.session.user.identifier
        path = os.path.join(self._steps_dir(user_id, thread_id), f"{step_id}.json")
        if os.path.exists(path):
            os.remove(path)

    async def get_thread_author(self, thread_id: str) -> str:
        user_id = self._find_thread_owner(thread_id)
        if user_id is None:
            raise ValueError(f"Thread {thread_id} not found")
        thread = self._load_thread(user_id, thread_id) or {}
        return thread.get("userIdentifier") or thread.get("userId") or user_id

    async def delete_thread(self, thread_id: str):
        user_id = self._find_thread_owner(thread_id)
        if user_id is None:
            return
        thread_dir = self._thread_dir(user_id, thread_id)
        if os.path.exists(thread_dir):
            shutil.rmtree(thread_dir)

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:
        user_id = filters.userId
        threads_root = os.path.join(self._user_dir(user_id), "threads")
        data: List[ThreadDict] = []
        if os.path.exists(threads_root):
            for thread_id in os.listdir(threads_root):
                thread = self._load_thread(user_id, thread_id)
                if thread is None:
                    continue
                data.append(
                    ThreadDict(
                        id=thread["id"],
                        createdAt=thread["createdAt"],
                        name=thread.get("name"),
                        userId=thread.get("userId"),
                        userIdentifier=thread.get("userIdentifier"),
                        metadata=thread.get("metadata"),
                        tags=thread.get("tags"),
                    ),
                )
        data.sort(key=lambda t: t.get("createdAt") or "", reverse=True)

        start = int(pagination.cursor) if pagination.cursor else 0
        end = start + pagination.first if pagination.first else len(data)
        paginated = data[start:end]
        has_next = end < len(data)
        end_cursor = str(end) if has_next else None
        return PaginatedResponse(
            data=paginated,
            pageInfo=PageInfo(
                hasNextPage=has_next,
                startCursor=str(start) if paginated else pagination.cursor,
                endCursor=end_cursor,
            ),
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        user_id = self._find_thread_owner(thread_id)
        if user_id is None:
            return None
        thread = self._load_thread(user_id, thread_id)
        if thread is None:
            return None
        thread_dict = ThreadDict(
            id=thread["id"],
            createdAt=thread["createdAt"],
            name=thread.get("name"),
            userId=thread.get("userId"),
            userIdentifier=thread.get("userIdentifier"),
            metadata=thread.get("metadata"),
            tags=thread.get("tags"),
            steps=self._load_steps(user_id, thread_id),
            elements=self._load_elements(user_id, thread_id),
        )
        return thread_dict

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        owner_id = user_id or self._find_thread_owner(thread_id)
        if owner_id is None:
            owner_id = context.session.user.identifier

        existing = self._load_thread(owner_id, thread_id) or {}
        item = {
            "id": thread_id,
            "createdAt": existing.get("createdAt", self._now()),
            "name": name if name is not None else existing.get("name"),
            "userId": owner_id,
            "userIdentifier": existing.get("userIdentifier", owner_id),
            "metadata": {**existing.get("metadata", {}), **(metadata or {})},
            "tags": tags if tags is not None else existing.get("tags"),
        }
        self._write_json(self._thread_path(owner_id, thread_id), item)

    async def delete_user_session(self, id: str) -> bool:
        return True

    async def close(self) -> None:
        return None
