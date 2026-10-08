"""推理参数路由：GET/PUT /api/params（技术方案 §3.3）。

范围/预设校验失败返回 400 INVALID_PARAMS；字段类型错误由 pydantic 转 422。
"""
from fastapi import APIRouter, HTTPException

from params import presets

router = APIRouter(prefix="/api/params", tags=["params"])


@router.get("")
def get_params():
    return presets.get()


@router.put("")
def put_params(patch: dict):
    try:
        return presets.save(patch)
    except ValueError as e:
        raise HTTPException(400, {"code": "INVALID_PARAMS",
                                  "message": str(e)})
