<script setup lang="ts">
// 调试抽屉「原始输出」：按到达顺序展示该消息的 SSE 原始帧（时间 / event / data 原文）
// 无帧记录时（如历史消息从后端加载）回退为拼接后的消息文本
import { computed } from 'vue'
import type { ChatMessage } from '../../api'
import { useDebugStore } from '../../stores/debug'

const props = defineProps<{ message: ChatMessage | null }>()

const debugStore = useDebugStore()
const frames = computed(() => (props.message ? debugStore.framesOf(props.message.id) : []))
const baseTime = computed(() => frames.value[0]?.time ?? 0)

// 相对首帧的毫秒偏移，直观看流式节奏
function offsetMs(time: number): string {
  return `+${Math.round((time - baseTime.value) * 1000)}ms`
}
</script>

<template>
  <div v-if="frames.length > 0" class="frame-list">
    <div v-for="(frame, i) in frames" :key="i" class="frame">
      <span class="frame-time">{{ offsetMs(frame.time) }}</span>
      <span class="frame-event" :class="frame.event">{{ frame.event }}</span>
      <span class="frame-data">{{ frame.data }}</span>
    </div>
  </div>
  <pre v-else class="raw-pane">{{ message?.content ?? '' }}</pre>
</template>

<style scoped>
.frame-list {
  font-family: ui-monospace, Consolas, monospace;
  font-size: 12px;
}
.frame {
  display: flex;
  gap: 8px;
  padding: 2px 0;
  border-bottom: 1px dashed #ebeef5;
}
.frame-time {
  color: #c0c4cc;
  min-width: 64px;
}
.frame-event {
  min-width: 48px;
  color: #909399;
}
.frame-event.token {
  color: #409eff;
}
.frame-event.error {
  color: #f56c6c;
}
.frame-data {
  flex: 1;
  word-break: break-all;
  white-space: pre-wrap;
  color: #606266;
}
.raw-pane {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 12px;
  color: #606266;
}
</style>
