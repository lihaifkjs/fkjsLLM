<script setup lang="ts">
// 调试抽屉「生效参数快照」（M5）：该消息生成时实际生效的推理参数（后端落库，非当前面板值）
// 数据源：observability 端点 params_snapshot；source 标注来源（全局默认值 / 用户已保存设置）
import { computed, toRef } from 'vue'
import type { ChatMessage } from '../../api'
import { useObservability } from '../../composables/useObservability'

const props = defineProps<{ message: ChatMessage | null }>()

const { observability } = useObservability(toRef(props, 'message'))

const snapshot = computed(() => observability.value?.params_snapshot ?? null)

// 预设与来源的中文标注（预设名对齐 ParamsView.vue）
const PRESET_LABEL: Record<string, string> = {
  precise: '精确',
  balanced: '平衡',
  creative: '创意',
  custom: '自定义',
}
const presetLabel = computed(() =>
  snapshot.value ? (PRESET_LABEL[snapshot.value.preset] ?? snapshot.value.preset) : '',
)
const sourceLabel = computed(() =>
  snapshot.value?.source === 'stored' ? '用户已保存设置' : '全局默认值',
)
</script>

<template>
  <el-descriptions v-if="snapshot" :column="1" border>
    <el-descriptions-item label="temperature">{{ snapshot.temperature }}</el-descriptions-item>
    <el-descriptions-item label="top_p">{{ snapshot.top_p }}</el-descriptions-item>
    <el-descriptions-item label="top_k">{{ snapshot.top_k }}</el-descriptions-item>
    <el-descriptions-item label="repetition_penalty">{{ snapshot.repetition_penalty }}</el-descriptions-item>
    <el-descriptions-item label="max_tokens">{{ snapshot.max_tokens }}</el-descriptions-item>
    <el-descriptions-item label="system_prompt">
      <span v-if="snapshot.system_prompt" class="system-prompt">{{ snapshot.system_prompt }}</span>
      <span v-else class="empty-value">（空）</span>
    </el-descriptions-item>
    <el-descriptions-item label="预设">{{ presetLabel }}（{{ snapshot.preset }}）</el-descriptions-item>
    <el-descriptions-item label="context_strategy">
      {{ snapshot.context_strategy === 'summary' ? '摘要压缩' : '滑窗硬截断' }}
    </el-descriptions-item>
    <el-descriptions-item label="来源">{{ sourceLabel }}</el-descriptions-item>
  </el-descriptions>
  <el-empty v-else description="该消息无参数快照（生成中或 M5 前的老消息）" />
</template>

<style scoped>
.system-prompt {
  white-space: pre-wrap;
  word-break: break-all;
}
.empty-value {
  color: #c0c4cc;
}
</style>
