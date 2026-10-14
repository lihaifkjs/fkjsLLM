"""模型加载开关路由：status / load / unload（PRD 3.8）。

M8：load 接受可选 model_id（注册表见 config.MODELS），切换模型 =
先 unload 再带 model_id load；status 下发当前 model_id 与可选条目。
"""
from fastapi import APIRouter
from pydantic import BaseModel

from config import MODELS, config
from model_manager.manager import model_manager

router = APIRouter(prefix="/api/model", tags=["model"])


class LoadRequest(BaseModel):
    model_id: str | None = None


@router.get("/status")
def status():
    s = model_manager.status()
    # n_ctx 随状态下发，前端据此算上下文占用预算（不再写死常量）
    return {"state": s.state.value, "vram_used_mb": s.vram_used_mb,
            "error": s.error, "generating": s.generating, "n_ctx": config.n_ctx,
            "model_id": s.model_id,
            "available_models": [{"id": k, "name": v["name"]}
                                 for k, v in MODELS.items()]}


@router.post("/load")
def load(req: LoadRequest | None = None):
    state = model_manager.load(req.model_id if req else None)
    return {"state": state.value}


@router.post("/unload")
def unload():
    state = model_manager.unload()
    return {"state": state.value}
