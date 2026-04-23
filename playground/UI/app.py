import atexit
import functools
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

# change current directory to the directory of this file for loading resources
os.chdir(os.path.dirname(__file__))

try:
    import chainlit as cl
    import chainlit.data as cl_data
    from chainlit.context import context

    print(
        "If UI is not started, please go to the folder playground/UI and run `chainlit run app.py` to start the UI",
    )
except Exception:
    raise Exception(
        "Package chainlit is required for using UI. Please install it manually by running: "
        "`pip install chainlit` and then run `chainlit run app.py`",
    )

repo_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(repo_path)
from taskweaver.app.app import TaskWeaverApp
from taskweaver.memory.attachment import AttachmentType
from taskweaver.memory.type_vars import RoleName
from taskweaver.module.event_emitter import PostEventType, RoundEventType, SessionEventHandlerBase
from taskweaver.session.session import Session
from local_chainlit_data_layer import LocalChainlitDataLayer

project_path = os.path.join(repo_path, "project")
workspace_path = os.path.join(project_path, "workspace")
os.environ.setdefault("CHAINLIT_AUTH_SECRET", "taskweaver-local-dev-auth-secret")
app = TaskWeaverApp(app_dir=project_path, use_local_uri=True)
atexit.register(app.stop)
app_session_dict: Dict[str, Session] = {}
session_map_path = os.path.join(project_path, "workspace", "ui_session_map.json")


cl_data._data_layer = LocalChainlitDataLayer(base_path=os.path.join(workspace_path, "chainlit_data"))


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if not username or not password:
        return None
    return cl.User(
        identifier=username,
        metadata={"provider": "credentials", "role": "user"},
    )


def load_session_map() -> Dict[str, str]:
    if not os.path.exists(session_map_path):
        return {}
    with open(session_map_path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_session_map(session_map: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(session_map_path), exist_ok=True)
    with open(session_map_path, "w") as f:
        json.dump(session_map, f)


def elem(name: str, cls: str = "", attr: Dict[str, str] = {}, **attr_dic: str):
    all_attr = {**attr, **attr_dic}
    if cls:
        all_attr.update({"class": cls})

    attr_str = ""
    if len(all_attr) > 0:
        attr_str += "".join(f' {k}="{v}"' for k, v in all_attr.items())

    def inner(*children: str):
        children_str = "".join(children)
        return f"<{name}{attr_str}>{children_str}</{name}>"

    return inner


def txt(content: str, br: bool = True):
    content = content.replace("<", "&lt;").replace(">", "&gt;")
    if br:
        content = content.replace("\n", "<br>")
    else:
        content = content.replace("\n", "&#10;")
    return content


div = functools.partial(elem, "div")
span = functools.partial(elem, "span")
blinking_cursor = span("tw-end-cursor")()


def file_display(files: List[Tuple[str, str]], session_cwd_path: str):
    elements: List[cl.Element] = []
    for file_name, file_path in files:
        # if image, no need to display as another file
        if file_path.endswith((".png", ".jpg", ".jpeg", ".gif")):
            image = cl.Image(
                name=file_path,
                display="inline",
                path=file_path if os.path.isabs(file_path) else os.path.join(session_cwd_path, file_path),
                size="large",
            )
            elements.append(image)
        elif file_path.endswith((".mp3", ".wav", ".flac")):
            audio = cl.Audio(
                name="converted_speech",
                display="inline",
                path=file_path if os.path.isabs(file_path) else os.path.join(session_cwd_path, file_path),
            )
            elements.append(audio)
        else:
            if file_path.endswith(".csv"):
                import pandas as pd

                data = (
                    pd.read_csv(file_path)
                    if os.path.isabs(file_path)
                    else pd.read_csv(os.path.join(session_cwd_path, file_path))
                )
                row_count = len(data)
                table = cl.Text(
                    name=file_path,
                    content=f"There are {row_count} in the data. The top {min(row_count, 5)} rows are:\n"
                    + data.head(n=5).to_markdown(),
                    display="inline",
                )
                elements.append(table)
            else:
                print(f"Unsupported file type: {file_name} for inline display.")
            # download files from plugin context
            file = cl.File(
                name=file_name,
                display="inline",
                path=file_path if os.path.isabs(file_path) else os.path.join(session_cwd_path, file_path),
            )
            elements.append(file)
    return elements


def extract_message_files(
    user_msg_content: str,
    artifact_paths: List[str],
) -> Tuple[str, List[Tuple[str, str]]]:
    files: List[Tuple[str, str]] = []
    for file_path in artifact_paths:
        file_name = os.path.basename(file_path)
        files.append((file_name, file_path))

    pattern = r"(!?)\[(.*?)\]\((.*?)\)"
    matches = re.findall(pattern, user_msg_content)
    for match in matches:
        img_prefix, file_name, file_path = match
        if "://" in file_path:
            if not is_link_clickable(file_path):
                user_msg_content = user_msg_content.replace(
                    f"{img_prefix}[{file_name}]({file_path})",
                    file_name,
                )
            continue
        files.append((file_name, file_path))
        user_msg_content = user_msg_content.replace(
            f"{img_prefix}[{file_name}]({file_path})",
            file_name,
        )
    return user_msg_content, files


async def replay_session_history(session: Session) -> None:
    for round in session.memory.conversation.rounds:
        await cl.Message(author="User", content=round.user_query).send()

        artifact_paths = [
            p
            for p in round.post_list
            for a in p.attachment_list
            if a.type == AttachmentType.artifact_paths
            for p in a.content
        ]

        for post in [p for p in round.post_list if p.send_to == "User"]:
            user_msg_content, files = extract_message_files(post.message, artifact_paths)
            elements = file_display(files, session.execution_cwd)
            await cl.Message(
                author="TaskWeaver",
                content=user_msg_content,
                elements=elements if len(elements) > 0 else None,
            ).send()


async def persist_thread_binding(session: Session, message: Optional[str] = None) -> None:
    user = cl.user_session.get("user")
    if user is None:
        return

    thread_id = context.session.thread_id
    if thread_id is None:
        return

    session_map = load_session_map()
    session_map[thread_id] = session.session_id
    save_session_map(session_map)

    existing_name = None
    data_layer = cl.data._data_layer
    if data_layer is not None:
        thread = await data_layer.get_thread(thread_id)
        if thread is not None:
            existing_name = thread.get("name")

        thread_name = existing_name or (message.strip()[:80] if message and message.strip() else f"Chat {thread_id[:8]}")
        await data_layer.update_thread(
            thread_id=thread_id,
            name=thread_name,
            user_id=user.identifier,
            metadata={"taskweaver_session_id": session.session_id},
        )


def is_link_clickable(url: str):
    if url:
        try:
            response = requests.get(url)
            # If the response status code is 200, the link is clickable
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    else:
        return False


class ChainLitMessageUpdater(SessionEventHandlerBase):
    def __init__(self, root_step: cl.Step):
        self.root_step = root_step
        self.reset_cur_step()
        self.suppress_blinking_cursor()

    def reset_cur_step(self):
        self.cur_step: Optional[cl.Step] = None
        self.cur_attachment_list: List[Tuple[str, AttachmentType, str, bool]] = []
        self.cur_post_status: str = "Updating"
        self.cur_send_to: RoleName = "Unknown"
        self.cur_message: str = ""
        self.cur_message_is_end: bool = False
        self.cur_message_sent: bool = False

    def suppress_blinking_cursor(self):
        cl.run_sync(self.root_step.stream_token(""))
        if self.cur_step is not None:
            cl.run_sync(self.cur_step.stream_token(""))

    def handle_round(
        self,
        type: RoundEventType,
        msg: str,
        extra: Any,
        round_id: str,
        **kwargs: Any,
    ):
        if type == RoundEventType.round_error:
            self.root_step.is_error = True
            self.root_step.output = msg
            cl.run_sync(self.root_step.update())

    def handle_post(
        self,
        type: PostEventType,
        msg: str,
        extra: Any,
        post_id: str,
        round_id: str,
        **kwargs: Any,
    ):
        if type == PostEventType.post_start:
            self.reset_cur_step()
            self.cur_step = cl.Step(name=extra["role"], show_input=True, root=False)
            cl.run_sync(self.cur_step.__aenter__())
        elif type == PostEventType.post_end:
            assert self.cur_step is not None
            content = self.format_post_body(True)
            cl.run_sync(self.cur_step.stream_token(content, True))
            cl.run_sync(self.cur_step.__aexit__(None, None, None))  # type: ignore
            self.reset_cur_step()
        elif type == PostEventType.post_error:
            pass
        elif type == PostEventType.post_attachment_update:
            assert self.cur_step is not None, "cur_step should not be None"
            id: str = extra["id"]
            a_type: AttachmentType = extra["type"]
            is_end: bool = extra["is_end"]
            # a_extra: Any = extra["extra"]
            if len(self.cur_attachment_list) == 0 or id != self.cur_attachment_list[-1][0]:
                self.cur_attachment_list.append((id, a_type, msg, is_end))

            else:
                prev_msg = self.cur_attachment_list[-1][2]
                self.cur_attachment_list[-1] = (id, a_type, prev_msg + msg, is_end)

        elif type == PostEventType.post_send_to_update:
            self.cur_send_to = extra["role"]
        elif type == PostEventType.post_message_update:
            self.cur_message += msg
            if extra["is_end"]:
                self.cur_message_is_end = True
        elif type == PostEventType.post_status_update:
            self.cur_post_status = msg

        if self.cur_step is not None:
            content = self.format_post_body(False)
            cl.run_sync(self.cur_step.stream_token(content, True))
            if self.cur_message_is_end and not self.cur_message_sent:
                self.cur_message_sent = True
                self.cur_step.elements = [
                    *(self.cur_step.elements or []),
                    cl.Text(
                        content=self.cur_message,
                        display="inline",
                    ),
                ]
                cl.run_sync(self.cur_step.update())
        self.suppress_blinking_cursor()

    def get_message_from_user(self, prompt: str, timeout: int = 120) -> Optional[str]:
        ask_user_msg = cl.AskUserMessage(content=prompt, author=" ", timeout=timeout)
        res = cl.run_sync(ask_user_msg.send())
        cl.run_sync(ask_user_msg.remove())
        if res is not None:
            res_msg = cl.Message.from_dict(res)
            msg_txt = res_msg.content
            cl.run_sync(res_msg.remove())
            return msg_txt
        return None

    def get_confirm_from_user(
        self,
        prompt: str,
        actions: List[Union[Tuple[str, str], str]],
        timeout: int = 120,
    ) -> Optional[str]:
        cl_actions: List[cl.Action] = []
        for arg_action in actions:
            if isinstance(arg_action, str):
                cl_actions.append(cl.Action(name=arg_action, value=arg_action))
            else:
                name, value = arg_action
                cl_actions.append(cl.Action(name=name, value=value))
        ask_user_msg = cl.AskActionMessage(content=prompt, actions=cl_actions, author=" ", timeout=timeout)
        res = cl.run_sync(ask_user_msg.send())
        cl.run_sync(ask_user_msg.remove())
        if res is not None:
            for action in cl_actions:
                if action.value == res["value"]:
                    return action.value
        return None

    def format_post_body(self, is_end: bool) -> str:
        content_chunks: List[str] = []

        for attachment in self.cur_attachment_list:
            a_type = attachment[1]

            # skip artifact paths always
            if a_type in [AttachmentType.artifact_paths]:
                continue

            # skip Python in final result
            if is_end and a_type in [AttachmentType.reply_content]:
                continue

            content_chunks.append(self.format_attachment(attachment))

        if self.cur_message != "":
            if self.cur_send_to == "Unknown":
                content_chunks.append("**Message**:")
            else:
                content_chunks.append(f"**Message To {self.cur_send_to}**:")

            if not self.cur_message_sent:
                content_chunks.append(
                    self.format_message(self.cur_message, self.cur_message_is_end),
                )

        if not is_end:
            content_chunks.append(
                div("tw-status")(
                    span("tw-status-updating")(
                        elem("svg", viewBox="22 22 44 44")(elem("circle")()),
                    ),
                    span("tw-status-msg")(txt(self.cur_post_status + "...")),
                ),
            )

        return "\n\n".join(content_chunks)

    def format_attachment(
        self,
        attachment: Tuple[str, AttachmentType, str, bool],
    ) -> str:
        id, a_type, msg, is_end = attachment
        header = div("tw-atta-header")(
            div("tw-atta-key")(
                " ".join([item.capitalize() for item in a_type.value.split("_")]),
            ),
            div("tw-atta-id")(id),
        )
        atta_cnt: List[str] = []

        if a_type in [AttachmentType.plan, AttachmentType.init_plan]:
            items: List[str] = []
            lines = msg.split("\n")
            for idx, row in enumerate(lines):
                item = row
                if "." in row and row.split(".")[0].isdigit():
                    item = row.split(".", 1)[1].strip()
                items.append(
                    div("tw-plan-item")(
                        div("tw-plan-idx")(str(idx + 1)),
                        div("tw-plan-cnt")(
                            txt(item),
                            blinking_cursor if not is_end and idx == len(lines) - 1 else "",
                        ),
                    ),
                )
            atta_cnt.append(div("tw-plan")(*items))
        elif a_type in [AttachmentType.execution_result]:
            atta_cnt.append(
                elem("pre", "tw-execution-result")(
                    elem("code")(txt(msg)),
                ),
            )
        elif a_type in [AttachmentType.reply_content]:
            atta_cnt.append(
                elem("pre", "tw-python", {"data-lang": "python"})(
                    elem("code", "language-python")(txt(msg, br=False)),
                ),
            )
        else:
            atta_cnt.append(txt(msg))
            if not is_end:
                atta_cnt.append(blinking_cursor)

        return div("tw-atta")(
            header,
            div("tw-atta-cnt")(*atta_cnt),
        )

    def format_message(self, message: str, is_end: bool) -> str:
        content = txt(message, br=False)
        begin_regex = re.compile(r"^```(\w*)$\n", re.MULTILINE)
        end_regex = re.compile(r"^```$\n?", re.MULTILINE)

        if not is_end:
            end_tag = " " + blinking_cursor
        else:
            end_tag = ""

        while True:
            start_label = begin_regex.search(content)
            if not start_label:
                break
            start_pos = content.index(start_label[0])
            lang_tag = start_label[1]
            content = "".join(
                [
                    content[:start_pos],
                    f'<pre data-lang="{lang_tag}"><code class="language-{lang_tag}">',
                    content[start_pos + len(start_label[0]) :],
                ],
            )

            end_pos = end_regex.search(content)
            if not end_pos:
                content += end_tag + "</code></pre>"
                end_tag = ""
                break
            end_pos_pos = content.index(end_pos[0])
            content = f"{content[:end_pos_pos]}</code></pre>{content[end_pos_pos + len(end_pos[0]):]}"

        content += end_tag
        return content


@cl.on_chat_start
async def start():
    user_session_id = context.session.thread_id or cl.user_session.get("id")
    session_map = load_session_map()
    taskweaver_session_id = session_map.get(user_session_id)
    session = app.get_session(session_id=taskweaver_session_id) if taskweaver_session_id is not None else app.get_session()
    session_map[user_session_id] = session.session_id
    save_session_map(session_map)
    app_session_dict[user_session_id] = session
    print(f"Starting session {session.session_id}")
    if taskweaver_session_id is not None:
        await replay_session_history(session)
    await persist_thread_binding(session)


@cl.on_chat_resume
async def on_chat_resume(thread: Dict[str, Any]):
    thread_id = thread["id"]
    metadata = thread.get("metadata") or {}
    taskweaver_session_id = metadata.get("taskweaver_session_id")

    if taskweaver_session_id is None:
        session_map = load_session_map()
        taskweaver_session_id = session_map.get(thread_id)

    session = app.get_session(session_id=taskweaver_session_id) if taskweaver_session_id is not None else app.get_session()
    app_session_dict[thread_id] = session
    await persist_thread_binding(session)


@cl.on_chat_end
async def end():
    user_session_id = context.session.thread_id or cl.user_session.get("id")
    app_session = app_session_dict[user_session_id]
    session_map = load_session_map()
    session_map[user_session_id] = app_session.session_id
    save_session_map(session_map)
    print(f"Detaching from session {app_session.session_id}")
    app_session_dict.pop(user_session_id)


@cl.on_message
async def main(message: cl.Message):
    user_session_id = context.session.thread_id or cl.user_session.get("id")  # type: ignore
    session: Session = app_session_dict[user_session_id]  # type: ignore
    session_cwd_path = session.execution_cwd
    await persist_thread_binding(session, message.content)

    # display loader before sending message
    async with cl.Step(name="", show_input=True, root=True) as root_step:
        response_round = await cl.make_async(session.send_message)(
            message.content,
            files=[
                {
                    "name": element.name if element.name else "file",
                    "path": element.path,
                }
                for element in message.elements
                if element.type == "file" or element.type == "image"
            ],
            event_handler=ChainLitMessageUpdater(root_step),
        )

    artifact_paths = [
        p
        for p in response_round.post_list
        for a in p.attachment_list
        if a.type == AttachmentType.artifact_paths
        for p in a.content
    ]

    for post in [p for p in response_round.post_list if p.send_to == "User"]:
        user_msg_content, files = extract_message_files(post.message, artifact_paths)
        elements = file_display(files, session_cwd_path)
        await cl.Message(
            author="TaskWeaver",
            content=f"{user_msg_content}",
            elements=elements if len(elements) > 0 else None,
        ).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
