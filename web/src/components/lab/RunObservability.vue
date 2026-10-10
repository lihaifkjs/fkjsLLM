<script setup lang="ts">
// 实验场单栏观测展开区（M6 计划 5.7）：三段式——token 统计（占比条样式复用 debug/TokensPane）、
// 性能（TTFT/prefill/decode + token/s）、参数快照（本组参数，来源标注「实验自定义」）
// 数据源：实验详情 run 字段（决策 B：随 GET /experiments/{id} 下发，不走 useObservability 通道）
import { computed } from 'vue'
import type { ExperimentRun } from '../../api'

const props = defineProps<{ run: ExperimentRun }>()

const hasTokenData = computed(() => props.run.input_tokens != null)

// 三段占比条：宽度按 input_tokens 占比（实验单轮 history 恒为 0，色块不渲染宽度）
interface Seg {
  key: 'system' | 'history' | 'current'
  label: string
  tokens: number
  percent: number
}
const segments = computed<Seg[]>(() => {
  const seg = props.run.prompt_seg_tokens
  const input = props.run.input_tokens
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

const hasTiming = computed(() => props.run.ttft_ms != null)

// 聚合输出速度：output_tokens / decode 耗时（秒换算），保留一位小数
const tokensPerSec = computed(() => {
  const { output_tokens, decode_ms } = props.run
  if (output_tokens == null || !decode_ms || decode_ms <= 0) return null
  return Math.round((output_tokens / (decode_ms / 1000)) * 10) / 10
})

function fmtMs(ms: number | null): string {
  return ms === null ? '—' : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`
}
</script>

<template>
  <div class="run-observability">
    <template v-if="hasTokenData">
      <div class="obs-title">Token 统计</div>
      <el-descriptions :column="2" border size="small">
        <el-descriptions-item label="输入 tokens">{{ run.input_tokens }}</el-descriptions-item>
        <el-descriptions-item label="输出 tokens">{{ run.output_tokens ?? '—' }}</el-descriptions-item>
        <el-descriptions-item v-if="run.prompt_eval_tokens != null" label="实际 prefill tokens" :span="2">
          {{ run.prompt_eval_tokens }}
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

    <template v-if="hasTiming">
      <div class="obs-title">性能指标</div>
      <el-descriptions :column="2" border size="small">
        <el-descriptions-item label="首 token 延迟（TTFT）">{{ fmtMs(run.ttft_ms) }}</el-descriptions-item>
        <el-descriptions-item label="prefill 耗时">{{ fmtMs(run.prefill_ms) }}</el-descriptions-item>
        <el-descriptions-item label="decode 耗时">{{ fmtMs(run.decode_ms) }}</el-descriptions-item>
        <el-descriptions-item label="输出速度">
          {{ tokensPerSec === null ? '—' : `${tokensPerSec} token/s` }}
        </el-descriptions-item>
      </el-descriptions>
    </template>

    <div class="obs-title">参数快照</div>
    <el-descriptions :column="2" border size="small">
      <el-descriptions-item label="temperature">{{ run.params.temperature }}</el-descriptions-item>
      <el-descriptions-item label="top_p">{{ run.params.top_p }}</el-descriptions-item>
      <el-descriptions-item label="top_k">{{ run.params.top_k }}</el-descriptions-item>
      <el-descriptions-item label="max_tokens">{{ run.params.max_tokens }}</el-descriptions-item>
      <el-descriptions-item label="来源" :span="2">实验自定义</el-descriptions-item>
    </el-descriptions>
    <div class="obs-hint">repetition_penalty 与 system prompt 取创建实验时的全局值（决策 D）</div>

    <el-empty
      v-if="!hasTokenData && !hasTiming"
      description="暂无观测数据（生成完成后可查）"
      :image-size="60"
    />
  </div>
</template>

<style scoped>
.run-observability {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.obs-title {
  font-size: 12px;
  font-weight: 600;
  color: #606266;
  margin-top: 4px;
}
.obs-hint {
  font-size: 12px;
  color: #909399;
}
.seg-section {
  margin-top: 4px;
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
