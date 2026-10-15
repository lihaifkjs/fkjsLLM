"""M10 简单登录门：配置文件写死凭据 + 内存 token（重启失效，需重新登录）。

AUTH_ENABLED=false（默认）时 require_auth 直接放行，行为与 M9 之前完全一致。
"""
import secrets

from fastapi import HTTPException, Request

from config import config

_tokens: set[str] = set()


def login(username: str, password: str) -> str | None:
    """校验配置凭据，成功签发随机 token 并登记；失败返回 None。"""
    if config.auth_users.get(username) != password:
        return None
    token = secrets.token_urlsafe(32)
    _tokens.add(token)
    return token


def require_auth(request: Request) -> None:
    """挂在各业务路由上的统一守卫：校验 Authorization: Bearer <token>。"""
    if not config.auth_enabled:
        return
    header = request.headers.get("authorization", "")
    token = header[7:] if header.startswith("Bearer ") else None
    if token not in _tokens:
        raise HTTPException(401, {"code": "UNAUTHORIZED",
                                  "message": "未登录或登录已失效，请重新登录"})
