<script setup lang="ts">
// 调试抽屉（PRD 3.7）：一期仅占位，标签页配置化。
// 约束：新增标签页只允许「加配置 + 写组件」，不改抽屉容器与消息列表结构。
import { computed, ref, watch, type Component } from 'vue'
import type { ChatMessage } from '../api'
import PlaceholderPane from './debug/PlaceholderPane.vue'
import RawPane from './debug/RawPane.vue'

interface DebugTab {
  key: string
  label: string
  component: Component
}

// 二期候选标签页（PRD 3.7），一期统一渲染占位组件
// 「原始输出」为协议外新增：展示消息原文（含 <think> 标签），与主界面折叠渲染对照
const TABS: DebugTab[] = [
  { key: 'raw', label: '原始输出', component: RawPane },
  { key: 'prompt', label: 'Prompt 预览', component: PlaceholderPane },
  { key: 'tokens', label: 'Token 统计', component: PlaceholderPane },
  { key: 'trace', label: '调用链', component: PlaceholderPane },
  { key: 'perf', label: '性能指标', component: PlaceholderPane },
  { key: 'params', label: '生效参数快照', component: PlaceholderPane },
]

const props = defineProps<{ visible: boolean; message: ChatMessage | null }>()
const emit = defineEmits<{ 'update:visible': [value: boolean] }>()

const drawerVisible = computed({
  get: () => props.visible,
  set: (value: boolean) => emit('update:visible', value),
})

const activeTab = ref(TABS[0]!.key)
watch(
  () => props.visible,
  (visible) => {
    if (visible) activeTab.value = TABS[0]!.key
  },
)
</script>

<template>
  <el-drawer v-model="drawerVisible" title="调试信息" size="480px">
    <el-tabs v-model="activeTab">
      <el-tab-pane v-for="tab in TABS" :key="tab.key" :label="tab.label" :name="tab.key">
        <component :is="tab.component" :title="tab.label" :message="message" />
      </el-tab-pane>
    </el-tabs>
  </el-drawer>
</template>
