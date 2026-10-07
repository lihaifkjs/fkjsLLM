// 解析助手消息中的 <think> 思考块（Qwen3 输出，后端原样透传，见 docs/后端实现.md §4.2）
// 流式期间 </think> 可能尚未到达：此时 thinkingDone=false，全部内容视为思考中
export interface ThinkSplit {
  thinking: string | null // 思考内容；无思考块时为 null
  thinkingDone: boolean // </think> 是否已到达
  answer: string // 正式回答（思考块之后的部分）
}

export function splitThink(content: string): ThinkSplit {
  const OPEN = '<think>'
  const CLOSE = '</think>'
  if (!content.startsWith(OPEN)) {
    return { thinking: null, thinkingDone: true, answer: content }
  }
  const closeIndex = content.indexOf(CLOSE)
  if (closeIndex === -1) {
    return { thinking: content.slice(OPEN.length), thinkingDone: false, answer: '' }
  }
  return {
    thinking: content.slice(OPEN.length, closeIndex),
    thinkingDone: true,
    answer: content.slice(closeIndex + CLOSE.length),
  }
}
