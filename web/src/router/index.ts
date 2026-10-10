import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import LabView from '../views/LabView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: ChatView },
    // M6 参数实验场（决策 A：独立路由，不改 ChatView 结构）
    { path: '/lab', component: LabView },
  ],
})
