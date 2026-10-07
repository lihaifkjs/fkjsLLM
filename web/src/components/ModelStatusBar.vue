<script setup lang="ts">
// 模型加载/卸载开关 + 状态展示（PRD 3.4/3.8）：加载/卸载中禁止重复操作
import { computed } from 'vue'
import { ElMessage } from 'element-plus'
import { useModelStore } from '../stores/model'
import type { ModelState } from '../api'
import type { ApiError } from '../api'

const modelStore = useModelStore()

const STATE_META: Record<ModelState, { label: string; tag: 'info' | 'warning' | 'success' | 'danger' }> = {
  unloaded: { label: '未加载', tag: 'info' },
  loading: { label: '加载中', tag: 'warning' },
  loaded: { label: '已加载', tag: 'success' },
  unloading: { label: '卸载中', tag: 'warning' },
  error: { label: '错误', tag: 'danger' },
}

const meta = computed(() => STATE_META[modelStore.state])

async function onToggle() {
  try {
    if (modelStore.loaded) await modelStore.unload()
    else await modelStore.load()
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '操作失败')
  }
}
</script>

<template>
  <div class="model-status-bar">
    <el-tag :type="meta.tag" disable-transitions>
      {{ meta.label }}
      <template v-if="modelStore.state === 'loading'">（首次加载约需数十秒，请耐心等待）</template>
    </el-tag>
    <span v-if="modelStore.status.vram_used_mb !== null" class="vram">
      显存 {{ modelStore.status.vram_used_mb }} MB
    </span>
    <span v-if="modelStore.state === 'error'" class="error-text">{{ modelStore.status.error }}</span>
    <el-button
      size="small"
      :type="modelStore.loaded ? 'danger' : 'primary'"
      :loading="modelStore.busy"
      :disabled="modelStore.busy"
      @click="onToggle"
    >
      {{ modelStore.loaded ? '卸载模型' : '加载模型' }}
    </el-button>
  </div>
</template>

<style scoped>
.model-status-bar {
  display: flex;
  align-items: center;
  gap: 12px;
}
.vram {
  color: #909399;
  font-size: 13px;
}
.error-text {
  color: #f56c6c;
  font-size: 13px;
}
</style>
