<script setup lang="ts">
// 对话主界面：桌面双栏（左侧会话列表），移动端单栏 + 抽屉式会话列表（M4，PRD 3.5）；
// 右侧参数抽屉（M3），输入框上方为轻量可观测状态行（PRD 3.4）
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { Menu } from '@element-plus/icons-vue'
import ChatInput from '../components/ChatInput.vue'
import DebugDrawer from '../components/DebugDrawer.vue'
import MessageBubble from '../components/MessageBubble.vue'
import ModelStatusBar from '../components/ModelStatusBar.vue'
import SessionList from '../components/SessionList.vue'
import ParamsView from './ParamsView.vue'
import { useChatStream } from '../composables/useChatStream'
import { useIsMobile } from '../composables/useIsMobile'
import { useModelStore } from '../stores/model'
import { useSessionStore } from '../stores/session'
import { useSettingsStore } from '../stores/settings'
import type { ApiError, ChatMessage } from '../api'

// 上下文占用达到该比例时提示「上下文将满」（PRD 3.4 轻量版）
const CTX_WARN_RATIO = 0.8

const modelStore = useModelStore()
const sessionStore = useSessionStore()
const settingsStore = useSettingsStore()
const { streaming, tokensPerSec, lastUsage, send, stop } = useChatStream()

const listRef = ref<HTMLElement>()
const debugVisible = ref(false)
const debugMessage = ref<ChatMessage | null>(null)
const paramsVisible = ref(false)

// 移动端（M4）：侧栏改为抽屉，由顶栏汉堡按钮打开；选中会话后抽屉自动关闭
const isMobile = useIsMobile()
const sessionsDrawerVisible = ref(false)

// 新 token / 新消息时滚到底部
watch(
  () => sessionStore.messages.map((m) => m.content.length),
  async () => {
    await nextTick()
    if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight
  },
  { deep: true },
)

// 切换会话后上一轮的 usage 不再属于当前视图
watch(
  () => sessionStore.sessionId,
  () => {
    lastUsage.value = null
  },
)

// 每轮 done 后的上下文占用：input_tokens / (n_ctx - max_tokens)；input_tokens 为后端近似值
const contextUsage = computed(() => {
  if (!lastUsage.value) return null
  const budget = settingsStore.contextBudget
  const input = lastUsage.value.input_tokens
  const ratio = budget > 0 ? input / budget : 0
  return { input, budget, ratio }
})

function openDebug(message: ChatMessage) {
  debugMessage.value = message
  debugVisible.value = true
}

onMounted(async () => {
  modelStore.startPolling()
  try {
    await sessionStore.loadSessions()
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '加载会话列表失败')
  }
  try {
    await settingsStore.load()
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '加载参数失败')
  }
})
onBeforeUnmount(() => modelStore.stopPolling())
</script>

<template>
  <el-container class="chat-view">
    <el-header class="header">
      <el-button
        v-if="isMobile"
        class="menu-btn"
        :icon="Menu"
        text
        @click="sessionsDrawerVisible = true"
      />
      <span class="title">{{ sessionStore.title }}</span>
      <div class="header-right">
        <el-button size="small" @click="paramsVisible = true">参数</el-button>
        <ModelStatusBar />
      </div>
    </el-header>
    <el-container class="body">
      <el-aside v-if="!isMobile" width="220px" class="sidebar">
        <SessionList :disabled="streaming" />
      </el-aside>
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
        <div class="status-line">
          <span v-if="streaming" class="tps">生成中 · {{ tokensPerSec.toFixed(1) }} token/s</span>
          <template v-else-if="contextUsage">
            <span :class="{ warn: contextUsage.ratio >= CTX_WARN_RATIO }">
              上下文占用 {{ contextUsage.input }} / {{ contextUsage.budget }} tokens（{{
                Math.round(contextUsage.ratio * 100)
              }}%）
            </span>
            <span v-if="contextUsage.ratio >= CTX_WARN_RATIO" class="warn">
              上下文将满，建议开新会话
            </span>
          </template>
        </div>
        <div class="input-area">
          <ChatInput :streaming="streaming" @send="send" @stop="stop" />
        </div>
      </el-main>
    </el-container>
    <el-drawer
      v-model="sessionsDrawerVisible"
      direction="ltr"
      size="260px"
      title="会话"
      class="sessions-drawer"
    >
      <SessionList :disabled="streaming" @selected="sessionsDrawerVisible = false" />
    </el-drawer>
    <DebugDrawer v-model:visible="debugVisible" :message="debugMessage" />
    <ParamsView v-model:visible="paramsVisible" />
  </el-container>
</template>

<style scoped>
.chat-view {
  /* dvh 随移动端软键盘/地址栏收缩，避免输入框被遮挡；vh 为旧浏览器兜底 */
  height: 100vh;
  height: 100dvh;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e4e7ed;
}
.menu-btn {
  margin-right: 4px;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}
.title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 600;
}
.body {
  height: calc(100vh - 60px);
  height: calc(100dvh - 60px);
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
.status-line {
  display: flex;
  gap: 12px;
  align-items: center;
  min-height: 20px;
  padding: 0 24px;
  font-size: 12px;
  color: #909399;
}
.status-line .warn {
  color: #e6a23c;
}
.input-area {
  border-top: 1px solid #e4e7ed;
  padding: 12px 24px;
  /* iPhone 底部横条安全区（配合 index.html viewport-fit=cover） */
  padding-bottom: calc(12px + env(safe-area-inset-bottom));
}
/* 断点与 useIsMobile.MOBILE_BREAKPOINT 一致 */
@media (max-width: 768px) {
  .header {
    padding: 0 12px;
  }
  .header-right {
    gap: 8px;
  }
  .message-list {
    padding: 12px;
  }
  .status-line {
    padding: 0 12px;
  }
  .input-area {
    padding: 8px 12px;
    padding-bottom: calc(8px + env(safe-area-inset-bottom));
  }
}
</style>
