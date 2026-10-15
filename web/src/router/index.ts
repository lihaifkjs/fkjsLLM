import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import KnowledgeView from '../views/KnowledgeView.vue'
import LoginView from '../views/LoginView.vue'
import { api } from '../api'
import { getToken } from '../utils/auth'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: ChatView },
    // M7：知识库管理页（上传/列表/删除）
    { path: '/knowledge', component: KnowledgeView },
    // M10：登录页（仅后端开启 AUTH_ENABLED 时会被守卫引导至此）
    { path: '/login', component: LoginView },
  ],
})

// M10 登录门守卫：后端未开 AUTH_ENABLED 直接放行；开启了且无 token 跳 /login。
// enabled 状态按页面加载缓存一次，后端重启翻转开关属于低频运维操作，刷新页面即重取。
let authEnabledCache: boolean | null = null

router.beforeEach(async (to) => {
  if (to.path === '/login') return true
  if (authEnabledCache === null) {
    try {
      authEnabledCache = (await api.getAuthStatus()).enabled
    } catch {
      authEnabledCache = false // 状态查询失败不阻塞使用
    }
  }
  if (!authEnabledCache) return true
  return getToken() ? true : '/login'
})
