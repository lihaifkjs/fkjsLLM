<script setup lang="ts">
// 调试抽屉「调用链」（M6.5 填实，PRD 二期 §3）：摘要压缩调用的完整记录
// 三态：本消息触发了新压缩（完整记录）/ 复用既有摘要 / 无压缩（含滑窗策略）
// M7 将在此追加检索调用记录；M9 追加工具调用记录
import { computed, toRef } from 'vue'
import type { ChatMessage, CompressionRecord } from '../../api'
import { useObservability } from '../../composables/useObservability'

const props = defineProps<{ message: ChatMessage | null }>()

const { observability, loading } = useObservability(toRef(props, 'message'))

const compression = computed(() => observability.value?.compression ?? null)
const record = computed<CompressionRecord | null>(() =>
  compression.value?.triggered ? compression.value : null,
)
// 复用分支的摘要 id（模板里 TS 无法从 record 为空反推三态，单独收窄）
const reuseId = computed(() =>
  compression.value && !compression.value.triggered ? compression.value.active_id : null,
)

// 压缩率 = 1 - 摘要/原文；source_tokens 缺失（老数据）时不展示
const ratio = computed(() => {
  const r = record.value
  if (!r?.source_tokens || !r.summary_tokens) return null
  return Math.round((1 - r.summary_tokens / r.source_tokens) * 100)
})

// 对比条：source_tokens 与 summary_tokens 的相对宽度（source 恒为 100%）
const barPercent = computed(() => {
  const r = record.value
  if (!r?.source_tokens || !r.summary_tokens) return null
  return Math.max(2, Math.round((r.summary_tokens / r.source_tokens) * 100))
})
</script>

<template>
  <div v-if="loading" class="hint">加载中…</div>

  <template v-else-if="record">
    <el-descriptions :column="1" border>
      <el-descriptions-item label="事件">本轮触发了摘要压缩（#{{ record.id }}）</el-descriptions-item>
      <el-descriptions-item v-if="record.source_tokens != null" label="压缩前 → 后">
        {{ record.source_tokens }} → {{ record.summary_tokens }} tokens
        <span v-if="ratio !== null" class="ratio-hint">压缩掉 {{ ratio }}%</span>
      </el-descriptions-item>
      <el-descriptions-item v-if="record.timing" label="压缩耗时">
        TTFT {{ Math.round(record.timing.ttft_ms) }}ms / prefill
        {{ Math.round(record.timing.prefill_ms) }}ms / decode
        {{ Math.round(record.timing.decode_ms) }}ms
      </el-descriptions-item>
      <el-descriptions-item label="覆盖范围">消息 id ≤ {{ record.covered_upto }}</el-descriptions-item>
    </el-descriptions>

    <div v-if="barPercent !== null" class="cmp-section">
      <div class="cmp-title">token 对比</div>
      <div class="cmp-row">
        <span class="cmp-label">原文</span>
        <div class="cmp-bar"><div class="cmp-part source" style="width: 100%" /></div>
      </div>
      <div class="cmp-row">
        <span class="cmp-label">摘要</span>
        <div class="cmp-bar"><div class="cmp-part summary" :style="{ width: `${barPercent}%` }" /></div>
      </div>
    </div>

    <el-collapse class="detail">
      <el-collapse-item title="产出摘要（进入后续上下文的内容）" name="summary">
        <div class="text-block">{{ record.summary }}</div>
      </el-collapse-item>
      <el-collapse-item
        v-if="record.prev_summary"
        title="既有摘要（本次压缩的增量输入）"
        name="prev"
      >
        <div class="text-block">{{ record.prev_summary }}</div>
      </el-collapse-item>
      <el-collapse-item title="压缩前的完整 prompt" name="prompt">
        <div class="text-block mono">{{ record.source_prompt }}</div>
      </el-collapse-item>
    </el-collapse>
  </template>

  <el-alert
    v-else-if="reuseId !== null"
    type="info"
    :closable="false"
    :title="`复用既有摘要 #${reuseId}（本轮未触发新压缩）`"
  />
  <el-empty v-else description="无压缩记录（滑窗策略、未超限或上下文不足触发条件）" />
</template>

<style scoped>
.hint {
  color: #909399;
  font-size: 13px;
}
.ratio-hint {
  margin-left: 8px;
  font-size: 12px;
  color: #67c23a;
}
.cmp-section {
  margin-top: 12px;
}
.cmp-title {
  font-size: 12px;
  color: #909399;
  margin-bottom: 6px;
}
.cmp-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}
.cmp-label {
  font-size: 12px;
  color: #606266;
  width: 32px;
  flex-shrink: 0;
}
.cmp-bar {
  flex: 1;
  height: 12px;
  border-radius: 3px;
  background: #f4f4f5;
  overflow: hidden;
}
.cmp-part {
  height: 100%;
}
.cmp-part.source {
  background: #409eff;
}
.cmp-part.summary {
  background: #67c23a;
}
.detail {
  margin-top: 12px;
}
.text-block {
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 13px;
  color: #606266;
}
.mono {
  font-family: ui-monospace, Consolas, monospace;
  font-size: 12px;
}
</style>
