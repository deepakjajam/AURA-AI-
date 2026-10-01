from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import current_user
from app.models import User
from app.services.inference import LocalQwenProvider

router = APIRouter()


class AuraOnlineInput(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@router.post("/online")
def aura_online(
    data: AuraOnlineInput,
    user: User = Depends(current_user),
):
    provider = LocalQwenProvider()

    return {
        "reply": provider.generate(data.message),
        "model": "aura-online-local",
        "user": user.email,
    }
