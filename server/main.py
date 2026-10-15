"""FastAPI 入口：挂载 api 路由与前端静态文件。

启动（开发）：cd server && uvicorn main:app --host 0.0.0.0 --port 8000
"""
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api import auth as auth_api
from api import chat as chat_api
from api import knowledge as knowledge_api
from api import model as model_api
from api import params as params_api
from api import sessions as sessions_api
from auth import require_auth
from model_manager.manager import ServiceError

app = FastAPI(title="fkjsLLM", docs_url="/api/docs")


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError):
    # 业务错误（非法状态迁移、生成冲突等）统一 409，见技术方案 3.4
    return JSONResponse(status_code=409,
                        content={"code": exc.code, "message": exc.message})


app.include_router(auth_api.router)
# M10：业务路由统一挂登录门守卫（AUTH_ENABLED=false 时守卫放行）
for _m in (chat_api, model_api, sessions_api, params_api, knowledge_api):
    app.include_router(_m.router, dependencies=[Depends(require_auth)])

# 生产：托管前端 build 产物（web 下 npm run build）；不存在则跳过（纯后端开发）
_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
