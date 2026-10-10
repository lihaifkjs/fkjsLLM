<script setup lang="ts">
// 参数实验场（M6，二期 PRD §4）：同一 prompt 用 2–3 组参数串行生成、多栏并排对比
// 布局：编辑态（prompt 输入 + 参数栏配置 + 开始对比）/ 结果态（多栏并排，<768px 纵向堆叠）；
// 历史实验抽屉（el-drawer）加载旧实验
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import ParamSetEditor from '../components/lab/ParamSetEditor.vue'
import RunColumn from '../components/lab/RunColumn.vue'
import ModelStatusBar from '../components/ModelStatusBar.vue'
import { useIsMobile } from '../composables/useIsMobile'
import { useLabStore } from '../stores/lab'
import { useModelStore } from '../stores/model'
import { validateParamFields } from '../utils/paramFields'
import type { ApiError, RunParams } from '../api'

// 参数组数量约束（契约：runs ×2~3）
const MIN_SETS = 2
const MAX_SETS = 3

// 默认两组：精确 / 创意（采样值对齐 presets.py 预设），便于直接看到对比差异
const DEFAULT_SETS: RunParams[] = [
  { temperature: 0.1, top_p: 0.5, top_k: 10, max_tokens: 1024 },
  { temperature: 1.0, top_p: 0.9, top_k: 40, max_tokens: 1024 },
]

const router = useRouter()
const labStore = useLabStore()
const modelStore = useModelStore()
const isMobile = useIsMobile()

// editing=true 显示配置区；创建或打开历史实验后进入结果态
const editing = ref(true)
const prompt = ref('')
const paramSets = ref<RunParams[]>(DEFAULT_SETS.map((s) => ({ ...s })))
const starting = ref(false)
const historyVisible = ref(false)

function addParamSet() {
  if (paramSets.value.length >= MAX_SETS) return
  // 新增组复制最后一组，便于微调对比
  paramSets.value.push({ ...paramSets.value[paramSets.value.length - 1]! })
}

function removeParamSet(index: number) {
  if (paramSets.value.length <= MIN_SETS) return
  paramSets.value.splice(index, 1)
}

async function startCompare() {
  if (!prompt.value.trim()) {
    ElMessage.error('请输入对比用的 prompt')
    return
  }
  for (const [i, params] of paramSets.value.entries()) {
    const error = validateParamFields(params)
    if (error) {
      ElMessage.error(`参数组 ${i + 1}：${error}`)
      return
    }
  }
  starting.value = true
  try {
    await labStore.createExperiment(prompt.value.trim(), paramSets.value)
    editing.value = false
    void labStore.startQueue()
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '创建实验失败')
  } finally {
    starting.value = false
  }
}

function newExperiment() {
  if (labStore.queueRunning) {
    ElMessage.warning('队列生成中，请先等待完成或停止当前组')
    return
  }
  labStore.clearCurrent()
  editing.value = true
}

async function openHistory(id: number) {
  if (labStore.queueRunning) {
    ElMessage.warning('队列生成中，暂不能切换实验')
    return
  }
  try {
    await labStore.openExperiment(id)
    editing.value = false
    historyVisible.value = false
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '加载实验失败')
  }
}

async function removeHistory(id: number) {
  try {
    await ElMessageBox.confirm('删除后该实验及其全部运行结果不可恢复', '删除实验', {
      type: 'warning',
    })
  } catch {
    return // 用户取消
  }
  try {
    await labStore.removeExperiment(id)
    if (!labStore.current) editing.value = true
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '删除失败')
  }
}

async function openHistoryDrawer() {
  historyVisible.value = true
  try {
    await labStore.loadExperiments()
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '加载历史实验失败')
  }
}

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString()
}

onMounted(() => {
  modelStore.startPolling()
})
onBeforeUnmount(() => modelStore.stopPolling())
</script>

<template>
  <el-container class="lab-view">
    <el-header class="header">
      <el-button :icon="ArrowLeft" text @click="router.push('/')">返回对话</el-button>
      <span class="title">参数实验场</span>
      <div class="header-right">
        <el-button size="small" @click="openHistoryDrawer">历史实验</el-button>
        <ModelStatusBar />
      </div>
    </el-header>

    <el-main class="main">
      <!-- 编辑态：prompt + 参数组配置 -->
      <template v-if="editing">
        <div class="prompt-area">
          <el-input
            v-model="prompt"
            type="textarea"
            :rows="3"
            placeholder="输入用于对比实验的 prompt（各参数组共用同一份输入）"
          />
        </div>
        <div class="param-sets">
          <ParamSetEditor
            v-for="(set, index) in paramSets"
            :key="index"
            :model-value="set"
            :index="index"
            :removable="paramSets.length > MIN_SETS"
            @update:model-value="(v: RunParams) => (paramSets[index] = v)"
            @remove="removeParamSet(index)"
          />
        </div>
        <div class="edit-actions">
          <el-button
            size="small"
            :disabled="paramSets.length >= MAX_SETS"
            @click="addParamSet"
          >
            添加参数组（{{ paramSets.length }}/{{ MAX_SETS }}）
          </el-button>
          <el-button
            type="primary"
            :loading="starting"
            :disabled="!modelStore.loaded"
            @click="startCompare"
          >
            开始对比
          </el-button>
          <span v-if="!modelStore.loaded" class="hint">需先加载模型</span>
        </div>
      </template>

      <!-- 结果态：多栏并排（<768px 纵向堆叠） -->
      <template v-else-if="labStore.current">
        <div class="result-header">
          <div class="result-prompt" :title="labStore.current.prompt">
            {{ labStore.current.prompt }}
          </div>
          <el-button size="small" :disabled="labStore.queueRunning" @click="newExperiment">
            新建实验
          </el-button>
        </div>
        <div class="run-columns">
          <RunColumn
            v-for="run in labStore.current.runs"
            :key="run.id"
            :run="run"
            @stop="labStore.stopRun(run.id)"
            @retry="labStore.retryRun(run.id)"
          />
        </div>
      </template>
    </el-main>

    <el-drawer
      v-model="historyVisible"
      title="历史实验"
      :size="isMobile ? '100%' : '380px'"
    >
      <el-empty v-if="labStore.experiments.length === 0" description="暂无历史实验" />
      <div v-else class="history-list">
        <div
          v-for="item in labStore.experiments"
          :key="item.id"
          class="history-item"
          @click="openHistory(item.id)"
        >
          <div class="history-prompt">{{ item.prompt }}</div>
          <div class="history-meta">
            <span>{{ item.run_count }} 组 · {{ formatTime(item.created_at) }}</span>
            <el-button
              link
              type="danger"
              size="small"
              @click.stop="removeHistory(item.id)"
            >
              删除
            </el-button>
          </div>
        </div>
      </div>
    </el-drawer>
  </el-container>
</template>

<style scoped>
.lab-view {
  height: 100vh;
  height: 100dvh;
}
.header {
  display: flex;
  align-items: center;
  border-bottom: 1px solid #e4e7ed;
  gap: 8px;
}
.title {
  flex: 1;
  min-width: 0;
  font-weight: 600;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}
.main {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px 24px;
  overflow-y: auto;
}
.param-sets {
  display: flex;
  gap: 12px;
  align-items: stretch;
}
.edit-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}
.hint {
  font-size: 12px;
  color: #909399;
}
.result-header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.result-prompt {
  flex: 1;
  min-width: 0;
  padding: 8px 12px;
  border-radius: 6px;
  background: #f4f4f5;
  font-size: 13px;
  white-space: pre-wrap;
  word-break: break-word;
  /* 长 prompt 限高内滚，避免把结果栏顶出视口 */
  max-height: 120px;
  overflow-y: auto;
}
.run-columns {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}
.history-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.history-item {
  padding: 10px 12px;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  cursor: pointer;
}
.history-item:hover {
  border-color: #409eff;
}
.history-prompt {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.history-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 4px;
  font-size: 12px;
  color: #909399;
}
/* 断点与 useIsMobile.MOBILE_BREAKPOINT 一致 */
@media (max-width: 768px) {
  .main {
    padding: 12px;
  }
  .param-sets,
  .run-columns {
    flex-direction: column;
  }
  .header-right {
    gap: 8px;
  }
}
</style>
