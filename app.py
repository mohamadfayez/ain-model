# Filename: app.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

app = FastAPI()  # <-- This must be named `app`

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "MBZUAI/AIN", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("MBZUAI/AIN")

class Request(BaseModel):
    messages: list

@app.post("/infer")
def infer(request: Request):
    messages = request.messages
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return {"result": output}
