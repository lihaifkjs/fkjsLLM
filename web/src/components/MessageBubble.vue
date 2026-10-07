<script setup lang="ts">
// 消息气泡：marked 渲染 Markdown，代码块用 highlight.js 高亮
// 高亮不走 marked 扩展，而是渲染后对 pre code 跑 hljs.highlightElement（流式期间随内容更新反复高亮）
import { computed, nextTick, ref, watch } from 'vue'
import { marked } from 'marked'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'
import type { ChatMessage } from '../api'

const props = defineProps<{ message: ChatMessage }>()
const emit = defineEmits<{ openDebug: [message: ChatMessage] }>()

const bodyRef = ref<HTMLElement>()

const rendered = computed(() => marked.parse(props.message.content, { async: false }))

watch(
  rendered,
  async () => {
    await nextTick()
    bodyRef.value?.querySelectorAll('pre code').forEach((el) => {
      const codeEl = el as HTMLElement
      // hljs 高亮后会打 data-highlighted 标记并跳过重高亮；流式期间内容在变，须清掉标记
      delete codeEl.dataset.highlighted
      hljs.highlightElement(codeEl)
    })
  },
  { immediate: true },
)
</script>

<template>
  <div class="bubble-row" :class="message.role">
    <div class="bubble">
      <div ref="bodyRef" class="markdown-body" v-html="rendered" />
      <div v-if="message.role === 'assistant'" class="bubble-actions">
        <el-button link size="small" @click="emit('openDebug', message)">调试</el-button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bubble-row {
  display: flex;
  margin: 8px 0;
}
.bubble-row.user {
  justify-content: flex-end;
}
.bubble {
  max-width: 78%;
  padding: 10px 14px;
  border-radius: 10px;
  background: #f4f4f5;
}
.bubble-row.user .bubble {
  background: #d9ecff;
}
.bubble-actions {
  text-align: right;
  margin-top: 2px;
}
.markdown-body :deep(pre) {
  background: #f6f8fa;
  padding: 10px;
  border-radius: 6px;
  overflow-x: auto;
}
.markdown-body :deep(code) {
  font-family: ui-monospace, Consolas, monospace;
  font-size: 0.9em;
}
.markdown-body :deep(p:first-child) {
  margin-top: 0;
}
.markdown-body :deep(p:last-child) {
  margin-bottom: 0;
}
</style>
