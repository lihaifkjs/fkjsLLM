<script setup lang="ts">
// 桌面对话主界面：左侧栏位置预留（M3 放会话列表）
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import ChatInput from '../components/ChatInput.vue'
import DebugDrawer from '../components/DebugDrawer.vue'
import MessageBubble from '../components/MessageBubble.vue'
import ModelStatusBar from '../components/ModelStatusBar.vue'
import { useChatStream } from '../composables/useChatStream'
import { useModelStore } from '../stores/model'
import { useSessionStore } from '../stores/session'
import type { ChatMessage } from '../api'

const modelStore = useModelStore()
const sessionStore = useSessionStore()
const { streaming, send, stop } = useChatStream()

const listRef = ref<HTMLElement>()
const debugVisible = ref(false)
const debugMessage = ref<ChatMessage | null>(null)

// 新 token / 新消息时滚到底部
watch(
  () => sessionStore.messages.map((m) => m.content.length),
  async () => {
    await nextTick()
    if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight
  },
  { deep: true },
)

function openDebug(message: ChatMessage) {
  debugMessage.value = message
  debugVisible.value = true
}

onMounted(() => modelStore.startPolling())
onBeforeUnmount(() => modelStore.stopPolling())
</script>

<template>
  <el-container class="chat-view">
    <el-header class="header">
      <span class="title">{{ sessionStore.title }}</span>
      <ModelStatusBar />
    </el-header>
    <el-container class="body">
      <!-- M3：会话列表 -->
      <el-aside width="220px" class="sidebar" />
      <el-main class="main">
        <div ref="listRef" class="message-list">
          <el-empty v-if="sessionStore.messages.length === 0" description="加载模型后开始对话" />
          <MessageBubble
            v-for="message in sessionStore.messages"
            :key="message.id"
            :message="message"
            @open-debug="openDebug"
          />
        </div>
        <div class="input-area">
          <ChatInput :streaming="streaming" @send="send" @stop="stop" />
        </div>
      </el-main>
    </el-container>
    <DebugDrawer v-model:visible="debugVisible" :message="debugMessage" />
  </el-container>
</template>

<style scoped>
.chat-view {
  height: 100vh;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e4e7ed;
}
.title {
  font-weight: 600;
}
.body {
  height: calc(100vh - 60px);
}
.sidebar {
  border-right: 1px solid #e4e7ed;
  background: #fafafa;
}
.main {
  display: flex;
  flex-direction: column;
  padding: 0;
}
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px 24px;
}
.input-area {
  border-top: 1px solid #e4e7ed;
  padding: 12px 24px;
}
</style>
