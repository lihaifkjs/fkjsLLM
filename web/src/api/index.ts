// 按 VITE_API_MODE 导出 mock 或真实 client，组件层零感知（默认 mock）
import type { LlmApi } from './types'
import { api as mockApi } from './mock'
import { api as realApi } from './client'

const mode = import.meta.env.VITE_API_MODE ?? 'mock'

export const api: LlmApi = mode === 'real' ? realApi : mockApi
export * from './types'
