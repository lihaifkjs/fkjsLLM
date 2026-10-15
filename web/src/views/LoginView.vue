<script setup lang="ts">
// M10 简单登录门：仅在后端 AUTH_ENABLED=true 时由路由守卫引导至此
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../api'
import { setToken } from '../utils/auth'

const router = useRouter()
const form = reactive({ username: '', password: '' })
const loading = ref(false)
const error = ref('')

async function submit() {
  if (!form.username || !form.password) return
  loading.value = true
  error.value = ''
  try {
    const { token } = await api.login(form.username, form.password)
    setToken(token)
    router.push('/')
  } catch (e) {
    error.value = (e as { message?: string }).message ?? '登录失败'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <el-card class="login-card">
      <h2 class="login-title">fkjsLLM</h2>
      <el-form @submit.prevent="submit">
        <el-form-item>
          <el-input v-model="form.username" placeholder="用户名" autofocus />
        </el-form-item>
        <el-form-item>
          <el-input
            v-model="form.password"
            type="password"
            show-password
            placeholder="密码"
            @keyup.enter="submit"
          />
        </el-form-item>
        <el-alert v-if="error" :title="error" type="error" :closable="false" class="login-error" />
        <el-button
          type="primary"
          class="login-btn"
          :loading="loading"
          :disabled="!form.username || !form.password"
          @click="submit"
        >
          登录
        </el-button>
      </el-form>
    </el-card>
  </div>
</template>

<style scoped>
.login-page {
  min-height: 100dvh;
  display: flex;
  align-items: center;
  justify-content: center;
}
.login-card {
  width: 320px;
}
.login-title {
  margin: 0 0 16px;
  text-align: center;
}
.login-error {
  margin-bottom: 12px;
}
.login-btn {
  width: 100%;
}
</style>
