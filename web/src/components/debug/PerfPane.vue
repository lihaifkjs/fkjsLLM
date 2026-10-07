<script setup lang="ts">
// 调试抽屉「性能指标」：首 token 延迟（TTFT）、全程耗时、输出速度（数据源：帧时间戳）
// 二期扩展位：prefill/decode 分段耗时需后端在帧里带数据，届时在此组件内加行
import { computed } from 'vue'
import type { ChatMessage } from '../../api'
import { useDebugStore } from '../../stores/debug'
import { frameStats } from '../../utils/frameStats'

const props = defineProps<{ message: ChatMessage | null }>()

const debugStore = useDebugStore()
const stats = computed(() =>
  frameStats(props.message ? debugStore.framesOf(props.message.id) : []),
)
const hasData = computed(() => stats.value.ttftMs !== null)

function fmtMs(ms: number | null): string {
  return ms === null ? '—' : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms} ms`
}
</script>

<template>
  <el-descriptions v-if="hasData" :column="1" border>
    <el-descriptions-item label="首 token 延迟（TTFT）">{{ fmtMs(stats.ttftMs) }}</el-descriptions-item>
    <el-descriptions-item label="全程耗时">{{ fmtMs(stats.totalMs) }}</el-descriptions-item>
    <el-descriptions-item label="输出速度">
      {{ stats.tokensPerSec === null ? '—' : `${stats.tokensPerSec} token/s` }}
    </el-descriptions-item>
  </el-descriptions>
  <el-empty v-else description="该消息无性能数据（帧记录仅在本次会话内可用）" />
</template>
