"""推理参数：默认值、内置预设、范围校验与读写（技术方案 §3.3）。

存 settings 表（JSON 字符串），重启不丢；改参数不触发模型重载（PRD 3.8）。
"""
import json
from dataclasses import fields

from config import SamplingParams
from session_store.sqlite_store import session_store

SETTINGS_KEY = "sampling_params"

# 内置预设（PRD 3.3）：只覆盖采样四参数，max_tokens/system_prompt 不动
PRESETS: dict[str, dict] = {
    "precise":  {"temperature": 0.1, "top_p": 0.5, "top_k": 10,
                 "repetition_penalty": 1.05},
    "balanced": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                 "repetition_penalty": 1.05},
    "creative": {"temperature": 1.0, "top_p": 0.9, "top_k": 40,
                 "repetition_penalty": 1.05},
}

_NUMERIC_FIELDS = [f.name for f in fields(SamplingParams)
                   if f.name != "system_prompt"]


def _validate(values: dict):
    """范围校验，非法值抛 ValueError（api 层转 400）。"""
    if values["temperature"] < 0:
        raise ValueError("temperature 必须 ≥ 0")
    if not 0 < values["top_p"] <= 1:
        raise ValueError("top_p 必须在 (0, 1] 区间")
    if not isinstance(values["top_k"], int) or values["top_k"] < 0:
        raise ValueError("top_k 必须是非负整数")
    if values["repetition_penalty"] <= 0:
        raise ValueError("repetition_penalty 必须 > 0")
    if not isinstance(values["max_tokens"], int) or values["max_tokens"] < 1:
        raise ValueError("max_tokens 必须是正整数")


def _match_preset(values: dict) -> str:
    """当前采样值命中哪个预设；都不命中为 custom。"""
    for name, preset in PRESETS.items():
        if all(values[k] == v for k, v in preset.items()):
            return name
    return "custom"


def get_params() -> SamplingParams:
    """当前生效参数（chat 每次生成前调用）。无存档时返回默认值。"""
    raw = session_store.get_setting(SETTINGS_KEY)
    if raw is None:
        return SamplingParams()
    data = json.loads(raw)
    return SamplingParams(**{f.name: data[f.name] for f in fields(SamplingParams)})


def get() -> dict:
    """GET /api/params 的响应体：参数全量 + 命中的预设名。"""
    params = get_params()
    values = {f.name: getattr(params, f.name) for f in fields(SamplingParams)}
    return {**values, "preset": _match_preset(values)}


def save(patch: dict) -> dict:
    """PUT /api/params：合并补丁（preset 服务端展开，显式字段优先）并落库。"""
    values = get()
    values.pop("preset")
    preset_name = patch.pop("preset", None)
    if preset_name is not None:
        if preset_name not in PRESETS:
            raise ValueError(
                f"未知预设 {preset_name}，可选：{sorted(PRESETS)}")
        values.update(PRESETS[preset_name])
    unknown = set(patch) - set(values)
    if unknown:
        raise ValueError(f"未知参数字段：{sorted(unknown)}")
    values.update(patch)
    _validate(values)
    preset = preset_name if preset_name and all(
        values[k] == v for k, v in PRESETS[preset_name].items()
    ) else _match_preset(values)
    session_store.set_setting(SETTINGS_KEY, json.dumps(
        values, ensure_ascii=False))
    return {**values, "preset": preset}
