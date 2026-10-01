from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import current_user
from app.models import User
from app.services.inference import get_provider

router = APIRouter()


class ChatInput(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@router.post("")
def chat(
    data: ChatInput,
    user: User = Depends(current_user),
):
    provider = get_provider()
    return {
        "reply": provider.generate(data.message),
        "model": provider.model_name,
        "user": user.email,
    }


@router.post("/stream")
def chat_stream(
    data: ChatInput,
    user: User = Depends(current_user),
):
    provider = get_provider()
    return StreamingResponse(
        provider.generate_stream(data.message),
        media_type="text/plain; charset=utf-8",
        headers={
            "X-AURA-Model": provider.model_name,
            "Cache-Control": "no-cache",
        },
    )
