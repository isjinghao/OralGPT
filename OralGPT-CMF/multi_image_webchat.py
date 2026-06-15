import os
from pathlib import Path
from typing import Any

import gradio as gr

from llamafactory.chat import ChatModel


MODEL_PATH = "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"
ADAPTER_PATH = "/root/autodl-tmp/saves/qwen3vl-4b-oralgpt-cmf-lora"
TEMPLATE = "qwen3_vl_nothink"
IMAGE_MAX_PIXELS = 131072


chat_model: ChatModel | None = None


def get_chat_model() -> ChatModel:
    global chat_model
    if chat_model is None:
        chat_model = ChatModel(
            dict(
                model_name_or_path=MODEL_PATH,
                adapter_name_or_path=ADAPTER_PATH,
                template=TEMPLATE,
                trust_remote_code=True,
                infer_backend="huggingface",
                image_max_pixels=IMAGE_MAX_PIXELS,
            )
        )
    return chat_model


def normalize_file_path(file_item: Any) -> str | None:
    if file_item is None:
        return None
    if isinstance(file_item, str):
        return file_item
    if isinstance(file_item, dict):
        return file_item.get("path") or file_item.get("name")
    path = getattr(file_item, "path", None) or getattr(file_item, "name", None)
    return str(path) if path else None


def collect_image_paths(files: list[Any] | None) -> list[str]:
    paths: list[str] = []
    for item in files or []:
        path = normalize_file_path(item)
        if path:
            paths.append(path)
    return paths


def to_llamafactory_messages(history: list[dict[str, str]], user_text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history:
        if item.get("role") in {"user", "assistant"} and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": user_text})
    return messages


def respond(
    user_text: str,
    files: list[Any] | None,
    chatbot: list[dict[str, str]] | None,
    state_history: list[dict[str, str]] | None,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    user_text = (user_text or "").strip()
    image_paths = collect_image_paths(files)
    history = state_history or []

    if not user_text:
        return chatbot or [], history, ""

    model = get_chat_model()
    messages = to_llamafactory_messages(history, user_text)
    response = model.chat(
        messages=messages,
        system=system_prompt.strip() or None,
        images=image_paths or None,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=int(max_new_tokens),
    )
    answer = response[0].response_text

    next_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    next_chatbot = (chatbot or []) + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]
    return next_chatbot, next_history, ""


def clear_chat() -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    return [], [], ""


def show_gallery(files: list[Any] | None) -> list[str]:
    return collect_image_paths(files)


CSS = """
.gradio-container { max-width: 1280px !important; }
#chatbot { min-height: 560px; }
"""


with gr.Blocks(title="OralCMF Qwen3-VL Multi-Image Chat", css=CSS) as demo:
    gr.Markdown("## OralCMF Qwen3-VL Multi-Image Chat")
    state_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat",
                type="messages",
                elem_id="chatbot",
                height=620,
                show_copy_button=True,
            )
            user_input = gr.Textbox(
                label="Question",
                lines=4,
                placeholder="Ask a question about the uploaded images or continue the conversation.",
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            image_files = gr.File(
                label="Images",
                file_count="multiple",
                file_types=["image"],
                type="filepath",
            )
            gallery = gr.Gallery(label="Uploaded Images", columns=2, height=360)
            system_prompt = gr.Textbox(label="System Prompt", lines=3, value="")
            temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top P")
            max_new_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max New Tokens")

    image_files.change(show_gallery, inputs=image_files, outputs=gallery)
    submit_btn.click(
        respond,
        inputs=[
            user_input,
            image_files,
            chatbot,
            state_history,
            system_prompt,
            temperature,
            top_p,
            max_new_tokens,
        ],
        outputs=[chatbot, state_history, user_input],
    )
    user_input.submit(
        respond,
        inputs=[
            user_input,
            image_files,
            chatbot,
            state_history,
            system_prompt,
            temperature,
            top_p,
            max_new_tokens,
        ],
        outputs=[chatbot, state_history, user_input],
    )
    clear_btn.click(clear_chat, outputs=[chatbot, state_history, user_input])


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "6006"))
    demo.queue().launch(server_name=server_name, server_port=server_port, share=False)
