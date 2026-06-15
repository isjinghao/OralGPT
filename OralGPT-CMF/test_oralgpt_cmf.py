import os

from llamafactory.chat import ChatModel

MODEL_PATH = os.getenv("MODEL_PATH", "OralGPT/OralGPT-CMF-v1")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "").strip() or None
IMAGE_PATH = os.getenv(
    "IMAGE_PATH",
    "/root/workspace/LLaMA-Factory/data/SH9HCMFdata/group1/CHENFANG/XR/XP.png",
)

model_args = dict(
    model_name_or_path=MODEL_PATH,
    template="qwen3_vl_nothink",
    trust_remote_code=True,
    infer_backend="huggingface",
    image_max_pixels=131072,
)
if ADAPTER_PATH:
    model_args["adapter_name_or_path"] = ADAPTER_PATH

chat_model = ChatModel(model_args)

messages = [
    {
        "role": "user",
        "content": "<image>\nPlease generate a comprehensive panoramic x-ray report based on the provided panoramic x-ray image."
    }
]

images = [
    IMAGE_PATH
]

response = chat_model.chat(messages, images=images, temperature=0.2, max_new_tokens=1024)
print(response[0].response_text)
