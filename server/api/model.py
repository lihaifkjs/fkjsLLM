"""模型加载开关路由：status / load / unload（PRD 3.8）。"""
from fastapi import APIRouter

from model_manager.manager import model_manager

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status")
def status():
    s = model_manager.status()
    return {"state": s.state.value, "vram_used_mb": s.vram_used_mb,
            "error": s.error, "generating": s.generating}


@router.post("/load")
def load():
    state = model_manager.load()
    return {"state": state.value}


@router.post("/unload")
def unload():
    state = model_manager.unload()
    return {"state": state.value}
