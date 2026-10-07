import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [{ path: '/', component: ChatView }],
})
