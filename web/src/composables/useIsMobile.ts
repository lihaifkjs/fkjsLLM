// 移动端断点侦测（M4）：JS 侧唯一口径；CSS 媒体查询无法引用 JS 常量，
// 各组件内 @media (max-width: 768px) 与本常量保持同一数值，改断点时两边同步
import { onBeforeUnmount, ref } from 'vue'

export const MOBILE_BREAKPOINT = 768

export function useIsMobile() {
  const query = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT}px)`)
  const isMobile = ref(query.matches)
  const update = (e: MediaQueryListEvent) => {
    isMobile.value = e.matches
  }
  query.addEventListener('change', update)
  onBeforeUnmount(() => query.removeEventListener('change', update))
  return isMobile
}
