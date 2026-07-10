from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)


class LoginResponse(BaseModel):
    session_id: str


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    username = req.username.strip()
    if not username:
        username = "默认用户"
    return LoginResponse(session_id=username)
