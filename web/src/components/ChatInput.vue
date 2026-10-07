<script setup lang="ts">
// 输入框：模型未加载/加载中时禁用并提示「请先加载模型」（PRD 3.8）；
// 流式期间发送键变为停止键
import { computed, ref } from 'vue'
import { useModelStore } from '../stores/model'

const props = defineProps<{ streaming: boolean }>()
const emit = defineEmits<{ send: [content: string]; stop: [] }>()

const modelStore = useModelStore()
const input = ref('')

// 仅模型未加载时禁用（PRD 3.8）；流式期间可编辑文本但发送键变为停止键
const inputDisabled = computed(() => !modelStore.loaded)
const placeholder = computed(() =>
  modelStore.loaded ? '输入消息，Enter 发送，Shift+Enter 换行' : '请先加载模型',
)

function submit() {
  const content = input.value.trim()
  if (!content || inputDisabled.value || props.streaming) return
  emit('send', content)
  input.value = ''
}
</script>

<template>
  <div class="chat-input">
    <el-input
      v-model="input"
      type="textarea"
      :autosize="{ minRows: 2, maxRows: 8 }"
      :disabled="inputDisabled"
      :placeholder="placeholder"
      @keydown.enter.exact.prevent="submit"
    />
    <el-button v-if="streaming" type="danger" @click="emit('stop')">停止生成</el-button>
    <el-button v-else type="primary" :disabled="inputDisabled || !input.trim()" @click="submit">发送</el-button>
  </div>
</template>

<style scoped>
.chat-input {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}
.chat-input .el-button {
  flex-shrink: 0;
}
</style>
