<script setup lang="ts">
// 调试抽屉「Prompt 预览」（M5）：展示该消息实际送入模型的完整输入
// 数据源：observability 端点（滑窗截断后的真实 prompt，历史消息可查）；
// 「按角色」用 prompt_messages 分角色着色，「原文」用 prompt_text 等宽展示（特殊 token 可见）
// 注意：prompt_messages 的 assistant content 是渲染前的原始全文（含 think），
// 而 Qwen3 模板会剥掉历史 assistant 的 think——按角色视图将其灰显折叠并标注，
// 忠实呈现「模型实际看不到这段」（见 docs/后端m5实现.md §1.1）
import { computed, ref, toRef } from 'vue'
import type { ChatMessage } from '../../api'
import { useObservability } from '../../composables/useObservability'
import { splitThink } from '../../utils/think'

const props = defineProps<{ message: ChatMessage | null }>()

const { observability } = useObservability(toRef(props, 'message'))

const view = ref<'roles' | 'raw'>('roles')

const promptText = computed(() => observability.value?.prompt_text ?? null)
// assistant 消息拆出 think 块；其余角色原样
const promptMessages = computed(() =>
  (observability.value?.prompt_messages ?? null)?.map((m) =>
    m.role === 'assistant' ? { ...m, think: splitThink(m.content) } : { ...m, think: null },
  ),
)
const hasData = computed(() =>
  view.value === 'roles' ? promptMessages.value != null : promptText.value !== null,
)
</script>

<template>
  <div class="prompt-pane">
    <el-radio-group v-model="view" size="small" class="view-switch">
      <el-radio-button value="roles">按角色</el-radio-button>
      <el-radio-button value="raw">原文</el-radio-button>
    </el-radio-group>

    <template v-if="hasData">
      <div v-if="view === 'roles'" class="role-list">
        <div v-for="(msg, i) in promptMessages" :key="i" class="role-block" :class="msg.role">
          <span class="role-tag" :class="msg.role">{{ msg.role }}</span>
          <!-- think 段：Qwen3 模板对历史 assistant 会剥离，灰显折叠以忠实呈现 -->
          <details v-if="msg.think?.thinking" class="think-block">
            <summary>think · 不进入上下文（模板已剥离）</summary>
            <pre class="think-content">{{ msg.think.thinking }}</pre>
          </details>
          <pre class="role-content">{{ msg.think ? msg.think.answer : msg.content }}</pre>
        </div>
      </div>
      <pre v-else class="prompt-text">{{ promptText }}</pre>
    </template>
    <el-empty v-else description="该消息无 Prompt 数据（生成中或 M5 前的老消息）" />
  </div>
</template>

<style scoped>
.view-switch {
  margin-bottom: 12px;
}
.role-block {
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  margin-bottom: 8px;
  overflow: hidden;
}
.role-tag {
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  font-family: ui-monospace, Consolas, monospace;
  color: #fff;
}
.role-tag.system {
  background: #909399;
}
.role-tag.user {
  background: #409eff;
}
.role-tag.assistant {
  background: #67c23a;
}
.role-block.system {
  background: #f4f4f5;
}
.role-block.user {
  background: #ecf5ff;
}
.role-block.assistant {
  background: #f0f9eb;
}
.role-content {
  margin: 0;
  padding: 8px;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 13px;
  color: #606266;
}
.think-block {
  margin: 6px 8px 0;
  border: 1px dashed #c0c4cc;
  border-radius: 4px;
  background: #f4f4f5;
}
.think-block summary {
  padding: 4px 8px;
  font-size: 12px;
  color: #909399;
  cursor: pointer;
  user-select: none;
}
.think-content {
  margin: 0;
  padding: 4px 8px 8px;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 12px;
  color: #a8abb2;
}
.prompt-text {
  margin: 0;
  padding: 8px;
  background: #f4f4f5;
  border-radius: 4px;
  white-space: pre-wrap;
  word-break: break-all;
  font-family: ui-monospace, Consolas, monospace;
  font-size: 12px;
  color: #606266;
}
</style>
