from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import requests

app = FastAPI(title="Chatbot Proxy API", version="1.1.0")

# Allow all origins for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class Message(BaseModel):
    role: str = Field(..., description="Message role: system|user|assistant")
    content: str = Field(..., description="Message content text")


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = Field(
        default="openai/gpt-4o-mini",
        description="OpenRouter model identifier"
    )
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    api_key: Optional[str] = Field(
        default=None,
        description="Optional OpenRouter API key. If omitted, server env OPENROUTER_API_KEY is used. You can also pass it via 'x-openrouter-key' header or Authorization: Bearer <key>."
    )
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Additional OpenRouter params")


class ChatResponse(BaseModel):
    reply: str
    model: str


@app.get("/test")
def test():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    x_openrouter_key: Optional[str] = Header(default=None, alias="x-openrouter-key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    # Prefer body key, then env var, then custom header, then Bearer header
    bearer_key = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer_key = authorization.split(" ", 1)[1].strip()

    api_key = (
        req.api_key
        or os.getenv("OPENROUTER_API_KEY")
        or x_openrouter_key
        or bearer_key
    )

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=(
                "OpenRouter API key not provided. Add 'api_key' in JSON body, or set 'OPENROUTER_API_KEY' env var, "
                "or send header 'x-openrouter-key: <key>' or 'Authorization: Bearer <key>'."
            ),
        )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # These headers are recommended by OpenRouter for routing/analytics
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:3000"),
        "X-Title": os.getenv("APP_NAME", "Vibe Chatbot"),
    }

    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": [m.dict() for m in req.messages],
        "temperature": req.temperature,
        "top_p": req.top_p,
    }

    if req.extra and isinstance(req.extra, dict):
        payload.update(req.extra)

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")

    if r.status_code >= 400:
        try:
            data = r.json()
        except Exception:
            data = {"error": r.text}
        raise HTTPException(status_code=r.status_code, detail=data)

    data = r.json()
    # Extract assistant message text
    try:
        reply = data["choices"][0]["message"]["content"].strip()
        used_model = data.get("model", req.model)
    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected response format from OpenRouter")

    return ChatResponse(reply=reply, model=used_model)
