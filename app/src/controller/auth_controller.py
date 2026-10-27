from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from ..service.auth_service import COOKIE_NAME, login_user, authenticate, logout, allow_login_attempt

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class LoginResponse(BaseModel):
    session_id: str


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest, request: Request, response: Response):
    if not allow_login_attempt(request.client.host if request.client else "unknown"):
        raise HTTPException(status_code=429, detail="登录尝试过于频繁，请一分钟后重试")
    username = req.username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="用户名不能为空")
    try:
        local = request.url.hostname in {"localhost", "127.0.0.1", "::1"} and request.client is not None and request.client.host in {"127.0.0.1", "::1"}
        token = login_user(username, req.password, allow_create=local)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    response.set_cookie(COOKIE_NAME, token, max_age=30 * 86400, httponly=True,
                        secure=request.url.scheme == "https", samesite="strict")
    return LoginResponse(session_id=username)


@router.get("/me", response_model=LoginResponse)
def current_user(request: Request):
    username = authenticate(request.cookies.get(COOKIE_NAME))
    if not username:
        raise HTTPException(status_code=401, detail="请先登录")
    return LoginResponse(session_id=username)


@router.post("/logout")
def logout_user(request: Request, response: Response):
    logout(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}
