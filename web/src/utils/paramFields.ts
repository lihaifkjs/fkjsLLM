// 推理参数滑块范围与校验：与 server/params/presets.py 逐字对齐
// ParamsView（全局参数面板）与实验场 ParamSetEditor（单组实验参数）共用，mock 层校验同口径

export const PARAM_RANGES = {
  temperature: { min: 0, max: 2, step: 0.05 },
  top_p: { min: 0, max: 1, step: 0.05 },
  top_k: { min: 0, max: 100, step: 1 },
  repetition_penalty: { min: 0, max: 2, step: 0.05 },
  max_tokens: { min: 1, max: 8192, step: 1 },
} as const

// 校验传入的字段（缺省字段不查），返回第一条错误信息；无错返回 null
// 错误文案与 presets.py _validate 一致
export function validateParamFields(values: {
  temperature?: number
  top_p?: number
  top_k?: number
  repetition_penalty?: number
  max_tokens?: number
}): string | null {
  if (values.temperature !== undefined && values.temperature < 0) return 'temperature 必须 ≥ 0'
  if (values.top_p !== undefined && !(values.top_p > 0 && values.top_p <= 1))
    return 'top_p 必须在 (0, 1] 区间'
  if (values.top_k !== undefined && (!Number.isInteger(values.top_k) || values.top_k < 0))
    return 'top_k 必须是非负整数'
  if (values.repetition_penalty !== undefined && values.repetition_penalty <= 0)
    return 'repetition_penalty 必须 > 0'
  if (
    values.max_tokens !== undefined &&
    (!Number.isInteger(values.max_tokens) || values.max_tokens < 1)
  )
    return 'max_tokens 必须是正整数'
  return null
}
