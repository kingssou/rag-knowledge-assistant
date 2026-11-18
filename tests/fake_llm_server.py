"""本地假模型服务：模拟 OpenAI /chat/completions，仅用于验证适配器链路。"""
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

ANSWER = "国内一线城市的住宿费标准为每晚不超过五百元。[1]"


class Msg(BaseModel):
    role: str
    content: str


class Req(BaseModel):
    model: str = "fake"
    messages: list[Msg] = []
    temperature: float = 0.1
    max_tokens: int = 600
    stream: bool = False
    response_format: dict | None = None


@app.post("/v1/chat/completions")
def chat(req: Req):
    if req.response_format and req.response_format.get("type") == "json_object":
        content = json.dumps({"answer": ANSWER, "sources": [1]}, ensure_ascii=False)
    else:
        content = ANSWER

    if req.stream:
        def gen():
            for i in range(0, len(content), 6):
                chunk = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": content[i:i + 6]}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            done = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
    }


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "fake", "object": "model"}]}
