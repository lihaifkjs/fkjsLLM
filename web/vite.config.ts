import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    // real 模式联调：/api 代理到后端（技术方案 §5.5）
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
