<script setup lang="ts">
// 知识库管理页（M7，PRD 二期 §6）：上传 / 列表 / 删除，无详情页（决策 D/K）
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft, Upload } from '@element-plus/icons-vue'
import type { UploadRequestOptions } from 'element-plus'
import { api, type ApiError, type KnowledgeDocument } from '../api'

const router = useRouter()
const documents = ref<KnowledgeDocument[]>([])
const loading = ref(false)
const uploading = ref(false)

async function load() {
  loading.value = true
  try {
    documents.value = await api.listDocuments()
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '加载文档列表失败')
  } finally {
    loading.value = false
  }
}

async function upload(options: UploadRequestOptions) {
  uploading.value = true
  try {
    await api.uploadDocument(options.file)
    ElMessage.success(`已上传 ${options.file.name}`)
    await load()
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '上传失败')
  } finally {
    uploading.value = false
  }
}

async function remove(doc: KnowledgeDocument) {
  try {
    await ElMessageBox.confirm(
      `删除「${doc.filename}」？其切块将一并删除（历史调用链快照不受影响）。`,
      '删除文档',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return // 用户取消
  }
  try {
    await api.deleteDocument(doc.id)
    documents.value = documents.value.filter((d) => d.id !== doc.id)
    ElMessage.success('已删除')
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '删除失败')
  }
}

function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleString()
}

onMounted(load)
</script>

<template>
  <div class="knowledge-view">
    <el-header class="header">
      <el-button :icon="ArrowLeft" text @click="router.push('/')">返回</el-button>
      <span class="title">知识库</span>
      <el-upload
        accept=".txt,.md"
        :show-file-list="false"
        :http-request="upload"
        :disabled="uploading"
      >
        <el-button type="primary" :icon="Upload" :loading="uploading">上传文档</el-button>
      </el-upload>
    </el-header>
    <el-main class="main">
      <el-table v-loading="loading" :data="documents" class="table">
        <el-table-column prop="filename" label="文件名" min-width="180" />
        <el-table-column prop="char_count" label="字符数" width="100" align="right" />
        <el-table-column prop="chunk_count" label="切块数" width="90" align="right" />
        <el-table-column label="上传时间" width="180">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="90" align="center">
          <template #default="{ row }">
            <el-button size="small" type="danger" text @click="remove(row)">删除</el-button>
          </template>
        </el-table-column>
        <template #empty>
          <el-empty description="暂无文档，上传 .txt / .md 文件开始使用知识库检索" />
        </template>
      </el-table>
    </el-main>
  </div>
</template>

<style scoped>
.knowledge-view {
  height: 100vh;
  height: 100dvh;
  display: flex;
  flex-direction: column;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e4e7ed;
}
.title {
  flex: 1;
  font-weight: 600;
  margin-left: 8px;
}
.main {
  flex: 1;
  overflow-y: auto;
}
.table {
  max-width: 860px;
  margin: 0 auto;
}
</style>
