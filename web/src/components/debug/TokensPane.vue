<script setup lang="ts">
// 调试抽屉「Token 统计」：展示本轮输入/输出 token 数（数据源：done 帧 usage，近似值）
// 二期扩展位：各段占比（system/历史/当前消息）需后端支持，届时在此组件内加行
import { computed } from 'vue'
import type { ChatMessage } from '../../api'
import { useDebugStore } from '../../stores/debug'
import { frameStats } from '../../utils/frameStats'

const props = defineProps<{ message: ChatMessage | null }>()

const debugStore = useDebugStore()
const stats = computed(() =>
  frameStats(props.message ? debugStore.framesOf(props.message.id) : []),
)
const hasData = computed(() => stats.value.inputTokens !== null || stats.value.outputTokens !== null)
</script>

<template>
  <el-descriptions v-if="hasData" :column="1" border>
    <el-descriptions-item label="输入 tokens（近似）">{{ stats.inputTokens ?? '—' }}</el-descriptions-item>
    <el-descriptions-item label="输出 tokens">{{ stats.outputTokens ?? '—' }}</el-descriptions-item>
    <el-descriptions-item label="token 帧数">{{ stats.tokenFrames }}</el-descriptions-item>
  </el-descriptions>
  <el-empty v-else description="该消息无统计数据（帧记录仅在本次会话内可用）" />
</template>
