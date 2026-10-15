"""M10 登录门路由：GET /api/auth/status、POST /api/auth/login。

status 无需鉴权（前端据此决定是否跳登录页）；login 失败 401 INVALID_CREDENTIALS。
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import auth
from config import config

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str
    password: str


@router.get("/status")
def status():
    return {"enabled": config.auth_enabled}


@router.post("/login")
def login(body: LoginBody):
    token = auth.login(body.username, body.password)
    if token is None:
        raise HTTPException(401, {"code": "INVALID_CREDENTIALS",
                                  "message": "用户名或密码错误"})
    return {"token": token}
