<script setup lang="ts">
// 会话列表（PRD 3.2）：新建/切换/重命名/删除；空标题拦截；生成中禁止切换，避免流式写入串会话
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Edit, Plus } from '@element-plus/icons-vue'
import { useSessionStore } from '../stores/session'
import type { ApiError, SessionSummary } from '../api'

const props = defineProps<{ disabled: boolean }>()
// 选中/新建成功后通知父级（移动端用于自动关闭会话抽屉，桌面端无副作用）
const emit = defineEmits<{ selected: [] }>()

const sessionStore = useSessionStore()

// 生成中切换/删除会让流式写入脱离当前视图，统一拦截
function guardStreaming(): boolean {
  if (props.disabled) {
    ElMessage.info('生成中，请先停止或等待生成完成')
    return true
  }
  return false
}

async function onCreate() {
  if (guardStreaming()) return
  try {
    await sessionStore.createSession()
    emit('selected')
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '新建会话失败')
  }
}

async function onSelect(id: number) {
  if (id === sessionStore.sessionId || guardStreaming()) return
  try {
    await sessionStore.openSession(id)
    emit('selected')
  } catch (e) {
    const err = e as ApiError
    if (err.code === 'SESSION_NOT_FOUND') {
      ElMessage.warning('会话不存在或已被删除，请新建会话')
      sessionStore.handleSessionGone(id)
    } else {
      ElMessage.error(err.message ?? '加载会话失败')
    }
  }
}

async function onRename(session: SessionSummary) {
  let value: string
  try {
    const result = await ElMessageBox.prompt('请输入新标题', '重命名会话', {
      inputValue: session.title,
      inputValidator: (v) => (v && v.trim() ? true : '标题不能为空'),
    })
    value = result.value.trim()
  } catch {
    return // 取消
  }
  if (value === session.title) return
  try {
    await sessionStore.rename(session.id, value)
  } catch (e) {
    const err = e as ApiError
    if (err.code === 'SESSION_NOT_FOUND') {
      ElMessage.warning('会话不存在或已被删除')
      sessionStore.handleSessionGone(session.id)
    } else {
      ElMessage.error(err.message ?? '重命名失败')
    }
  }
}

async function onDelete(session: SessionSummary) {
  if (guardStreaming()) return
  try {
    await ElMessageBox.confirm(
      `删除会话「${session.title}」？该会话的全部消息将一并删除。`,
      '删除会话',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return // 取消
  }
  try {
    await sessionStore.remove(session.id)
  } catch (e) {
    const err = e as ApiError
    if (err.code === 'SESSION_NOT_FOUND') {
      ElMessage.warning('会话不存在或已被删除')
      sessionStore.handleSessionGone(session.id)
    } else {
      ElMessage.error(err.message ?? '删除失败')
    }
  }
}

function fmtTime(epochSec: number): string {
  const d = new Date(epochSec * 1000)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
</script>

<template>
  <div class="session-list">
    <el-button class="new-btn" type="primary" plain :icon="Plus" @click="onCreate">
      新建会话
    </el-button>
    <el-empty v-if="sessionStore.sessions.length === 0" description="暂无会话" :image-size="60" />
    <div
      v-for="session in sessionStore.sessions"
      :key="session.id"
      class="session-item"
      :class="{ active: session.id === sessionStore.sessionId }"
      @click="onSelect(session.id)"
    >
      <div class="session-text">
        <span class="session-title">{{ session.title }}</span>
        <span class="session-time">{{ fmtTime(session.updated_at) }}</span>
      </div>
      <span class="session-actions" @click.stop>
        <el-icon title="重命名" @click="onRename(session)"><Edit /></el-icon>
        <el-icon title="删除" @click="onDelete(session)"><Delete /></el-icon>
      </span>
    </div>
  </div>
</template>

<style scoped>
.session-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
  padding: 8px;
}
.new-btn {
  margin-bottom: 8px;
}
.session-item {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px;
  border-radius: 6px;
  cursor: pointer;
}
.session-item:hover {
  background: #f0f2f5;
}
.session-item.active {
  background: #ecf5ff;
}
.session-text {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}
.session-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 14px;
}
.session-time {
  color: #909399;
  font-size: 12px;
}
.session-actions {
  display: none;
  gap: 6px;
  color: #909399;
}
.session-item:hover .session-actions {
  display: flex;
}
/* 移动端无 hover，操作图标常显（断点与 useIsMobile.MOBILE_BREAKPOINT 一致） */
@media (max-width: 768px) {
  .session-actions {
    display: flex;
  }
}
.session-actions .el-icon:hover {
  color: #409eff;
}
</style>
