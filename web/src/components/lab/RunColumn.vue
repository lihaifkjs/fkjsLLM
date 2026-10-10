<script setup lang="ts">
// 实验场结果单栏（M6 计划 5.6）：状态徽标 + 流式文本 + 停止/重试 + 底部观测折叠区
import { computed, nextTick, ref, watch } from 'vue'
import type { ExperimentRun, RunStatus } from '../../api'
import RunObservability from './RunObservability.vue'

const props = defineProps<{ run: ExperimentRun }>()
const emit = defineEmits<{ stop: []; retry: [] }>()

// 状态 → 徽标文案与颜色（计划 5.3 状态映射）
const STATUS_META: Record<RunStatus, { label: string; type: 'info' | 'primary' | 'success' | 'danger' | 'warning' }> = {
  pending: { label: '排队中', type: 'info' },
  running: { label: '生成中', type: 'primary' },
  done: { label: '完成', type: 'success' },
  error: { label: '失败', type: 'danger' },
  stopped: { label: '已停止', type: 'warning' },
}
const statusMeta = computed(() => STATUS_META[props.run.status])

const canStop = computed(() => props.run.status === 'running')
const canRetry = computed(() => props.run.status === 'error' || props.run.status === 'stopped')

// 流式期间自动滚到底部
const contentRef = ref<HTMLElement>()
watch(
  () => props.run.content.length,
  async () => {
    await nextTick()
    if (contentRef.value) contentRef.value.scrollTop = contentRef.value.scrollHeight
  },
)

const obsOpen = ref(false)
// 观测数据是否已回填（done/stopped/error 后详情接口下发）
const hasObservability = computed(() => props.run.input_tokens != null)
</script>

<template>
  <div class="run-column">
    <div class="run-header">
      <span class="run-title">参数组 {{ run.seq + 1 }}</span>
      <el-tag size="small" :type="statusMeta.type" disable-transitions>
        {{ statusMeta.label }}
      </el-tag>
    </div>
    <div class="run-params">
      temperature {{ run.params.temperature }} · top_p {{ run.params.top_p }} · top_k
      {{ run.params.top_k }} · max_tokens {{ run.params.max_tokens }}
    </div>

    <div ref="contentRef" class="run-content">
      <template v-if="run.content">{{ run.content }}</template>
      <span v-else-if="run.status === 'running'" class="placeholder">等待首个 token…</span>
      <span v-else-if="run.status === 'pending'" class="placeholder">排队中，等待前一组完成</span>
      <span v-else class="placeholder">（无内容）</span>
    </div>

    <div v-if="run.status === 'error' && run.error" class="run-error">
      {{ run.error.code }}：{{ run.error.message }}
    </div>

    <div class="run-actions">
      <el-button v-if="canStop" size="small" type="danger" plain @click="emit('stop')">
        停止
      </el-button>
      <el-button v-if="canRetry" size="small" type="primary" plain @click="emit('retry')">
        重试本组
      </el-button>
      <el-button
        size="small"
        link
        :disabled="!hasObservability"
        @click="obsOpen = !obsOpen"
      >
        观测数据 {{ obsOpen ? '▾' : '▸' }}
      </el-button>
    </div>

    <div v-show="obsOpen && hasObservability" class="run-obs">
      <RunObservability :run="run" />
    </div>
  </div>
</template>

<style scoped>
.run-column {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
}
.run-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.run-title {
  font-size: 13px;
  font-weight: 600;
}
.run-params {
  font-size: 12px;
  color: #909399;
}
.run-content {
  min-height: 120px;
  max-height: 320px;
  overflow-y: auto;
  padding: 8px;
  border-radius: 6px;
  background: #f4f4f5;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}
.placeholder {
  color: #c0c4cc;
}
.run-error {
  font-size: 12px;
  color: #f56c6c;
}
.run-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}
.run-obs {
  border-top: 1px dashed #e4e7ed;
  padding-top: 8px;
}
</style>
