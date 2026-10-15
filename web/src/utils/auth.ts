// M10 简单登录门：token 本地存取与 401 跳转（leaf 模块，不依赖 api 层，避免循环引用）
const LS_TOKEN = 'fkjsllm.auth.token'

export function getToken(): string | null {
  return localStorage.getItem(LS_TOKEN)
}

export function setToken(token: string): void {
  localStorage.setItem(LS_TOKEN, token)
}

export function clearToken(): void {
  localStorage.removeItem(LS_TOKEN)
}

export function authHeader(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

// 401 时清 token 并跳登录页（已在登录页则不重复跳，避免死循环）
export function redirectToLogin(): void {
  clearToken()
  if (window.location.pathname !== '/login') window.location.href = '/login'
}
