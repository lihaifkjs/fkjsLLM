<script setup lang="ts">
// 实验场单组参数编辑器（M6 计划 5.5）：temperature / top_p / top_k / max_tokens 四滑块
// 范围与校验复用 utils/paramFields（与 server/params/presets.py 逐字对齐，ParamsView 同源）
import { computed } from 'vue'
import { PARAM_RANGES } from '../../utils/paramFields'
import type { RunParams } from '../../api'

const props = defineProps<{
  modelValue: RunParams
  index: number // 栏位序号（0 起），标题展示为「参数组 N」
  removable: boolean // 2–3 栏约束：仅 3 栏时可删
}>()
const emit = defineEmits<{
  'update:modelValue': [value: RunParams]
  remove: []
}>()

function patch(field: keyof RunParams, value: number) {
  emit('update:modelValue', { ...props.modelValue, [field]: value })
}

// el-slider 的 v-model 需要可写引用，逐字段包一层 computed
const temperature = computed({
  get: () => props.modelValue.temperature,
  set: (v: number) => patch('temperature', v),
})
const topP = computed({
  get: () => props.modelValue.top_p,
  set: (v: number) => patch('top_p', v),
})
const topK = computed({
  get: () => props.modelValue.top_k,
  set: (v: number) => patch('top_k', v),
})
const maxTokens = computed({
  get: () => props.modelValue.max_tokens,
  set: (v: number) => patch('max_tokens', v),
})
</script>

<template>
  <div class="param-set">
    <div class="param-set-header">
      <span class="param-set-title">参数组 {{ index + 1 }}</span>
      <el-button v-if="removable" link type="danger" size="small" @click="emit('remove')">
        删除
      </el-button>
    </div>
    <div class="field">
      <span class="field-label">temperature（≥ 0）</span>
      <el-slider
        v-model="temperature"
        :min="PARAM_RANGES.temperature.min"
        :max="PARAM_RANGES.temperature.max"
        :step="PARAM_RANGES.temperature.step"
        show-input
      />
    </div>
    <div class="field">
      <span class="field-label">top_p（0, 1]</span>
      <el-slider
        v-model="topP"
        :min="PARAM_RANGES.top_p.min"
        :max="PARAM_RANGES.top_p.max"
        :step="PARAM_RANGES.top_p.step"
        show-input
      />
    </div>
    <div class="field">
      <span class="field-label">top_k（非负整数）</span>
      <el-slider
        v-model="topK"
        :min="PARAM_RANGES.top_k.min"
        :max="PARAM_RANGES.top_k.max"
        :step="PARAM_RANGES.top_k.step"
        show-input
      />
    </div>
    <div class="field">
      <span class="field-label">max_tokens（正整数）</span>
      <el-slider
        v-model="maxTokens"
        :min="PARAM_RANGES.max_tokens.min"
        :max="PARAM_RANGES.max_tokens.max"
        :step="PARAM_RANGES.max_tokens.step"
        show-input
      />
    </div>
  </div>
</template>

<style scoped>
.param-set {
  flex: 1;
  min-width: 0;
  padding: 12px;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.param-set-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.param-set-title {
  font-size: 13px;
  font-weight: 600;
}
.field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.field-label {
  font-size: 13px;
  color: #606266;
}
</style>
