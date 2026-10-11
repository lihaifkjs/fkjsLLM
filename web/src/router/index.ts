import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import KnowledgeView from '../views/KnowledgeView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: ChatView },
    // M7：知识库管理页（上传/列表/删除）
    { path: '/knowledge', component: KnowledgeView },
  ],
})
