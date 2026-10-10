<script setup lang="ts">
// 调试抽屉「Token 统计」（M5）：输入/输出精确值 + 三段占比条（system/历史/当前）+ prompt_eval_tokens
// 数据源：observability 端点（历史消息可查）；取数失败或字段为 null（生成中未落库）时降级到帧统计
import { computed, toRef } from 'vue'
import type { ChatMessage } from '../../api'
import { useObservability } from '../../composables/useObservability'
import { useDebugStore } from '../../stores/debug'
import { frameStats } from '../../utils/frameStats'

const props = defineProps<{ message: ChatMessage | null }>()

const { observability } = useObservability(toRef(props, 'message'))
const debugStore = useDebugStore()

// 帧统计降级路径：生成中消息（观测尚未落库）仍展示 done 帧 usage 与 token 帧数
const stats = computed(() =>
  frameStats(props.message ? debugStore.framesOf(props.message.id) : []),
)

const obs = computed(() => observability.value)
const hasServerData = computed(() => obs.value?.input_tokens != null || obs.value?.output_tokens != null)

// 三段占比条：宽度按 input_tokens 占比；任一段为 0 时不渲染该色块
interface Seg {
  key: 'system' | 'history' | 'current'
  label: string
  tokens: number
  percent: number
}
const segments = computed<Seg[]>(() => {
  const seg = obs.value?.seg_tokens
  const input = obs.value?.input_tokens
  if (!seg || !input) return []
  const defs: { key: Seg['key']; label: string }[] = [
    { key: 'system', label: 'system' },
    { key: 'history', label: '历史' },
    { key: 'current', label: '当前' },
  ]
  return defs.map(({ key, label }) => ({
    key,
    label,
    tokens: seg[key],
    percent: Math.round((seg[key] / input) * 100),
  }))
})

// KV 前缀复用教学点：实际 prefill 算力小于账单口径时提示
const kvReused = computed(() => {
  const o = obs.value
  return o?.prompt_eval_tokens != null && o.input_tokens != null && o.prompt_eval_tokens < o.input_tokens
})
</script>

<template>
  <template v-if="hasServerData">
    <el-descriptions :column="1" border>
      <el-descriptions-item label="输入 tokens">{{ obs!.input_tokens ?? '—' }}</el-descriptions-item>
      <el-descriptions-item label="输出 tokens">{{ obs!.output_tokens ?? '—' }}</el-descriptions-item>
      <el-descriptions-item v-if="obs!.prompt_eval_tokens != null" label="实际 prefill tokens">
        {{ obs!.prompt_eval_tokens }}
        <span v-if="kvReused" class="kv-hint">
          KV 前缀复用生效：历史部分未重复计算（账单口径 {{ obs!.input_tokens }}）
        </span>
      </el-descriptions-item>
    </el-descriptions>

    <div v-if="segments.length > 0" class="seg-section">
      <div class="seg-title">输入构成（三段之和 = 输入 tokens）</div>
      <div class="seg-bar">
        <div
          v-for="seg in segments"
          :key="seg.key"
          class="seg-part"
          :class="seg.key"
          :style="{ width: `${seg.percent}%` }"
        />
      </div>
      <div class="seg-legend">
        <span v-for="seg in segments" :key="seg.key" class="seg-item">
          <i class="seg-dot" :class="seg.key" />{{ seg.label }} {{ seg.tokens }}（{{ seg.percent }}%）
        </span>
      </div>
    </div>
  </template>

  <el-descriptions v-else-if="stats.inputTokens !== null || stats.outputTokens !== null" :column="1" border>
    <el-descriptions-item label="输入 tokens">{{ stats.inputTokens ?? '—' }}</el-descriptions-item>
    <el-descriptions-item label="输出 tokens">{{ stats.outputTokens ?? '—' }}</el-descriptions-item>
    <el-descriptions-item label="token 帧数">{{ stats.tokenFrames }}</el-descriptions-item>
  </el-descriptions>
  <el-empty v-else description="该消息无统计数据（帧记录仅在本次会话内可用）" />
</template>

<style scoped>
.kv-hint {
  margin-left: 8px;
  font-size: 12px;
  color: #e6a23c;
}
.seg-section {
  margin-top: 12px;
}
.seg-title {
  font-size: 12px;
  color: #909399;
  margin-bottom: 6px;
}
.seg-bar {
  display: flex;
  height: 12px;
  border-radius: 3px;
  overflow: hidden;
  background: #f4f4f5;
}
.seg-part.system {
  background: #909399;
}
.seg-part.history {
  background: #409eff;
}
.seg-part.current {
  background: #67c23a;
}
.seg-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 6px;
  font-size: 12px;
  color: #606266;
}
.seg-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.seg-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 2px;
}
.seg-dot.system {
  background: #909399;
}
.seg-dot.history {
  background: #409eff;
}
.seg-dot.current {
  background: #67c23a;
}
</style>
