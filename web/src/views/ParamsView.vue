<script setup lang="ts">
// 参数面板（PRD 3.3）：右侧抽屉，5 个采样参数 + system prompt + 三预设
// 契约见 docs/后端m3实现.md §2.2：PUT 为部分字段补丁，预设服务端展开，校验失败 400 INVALID_PARAMS
// 前端校验规则与 server/params/presets.py 保持一致
import { computed, reactive, ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { useSettingsStore } from '../stores/settings'
import { useIsMobile } from '../composables/useIsMobile'
import type { ApiError, InferenceParams } from '../api'

// 预设只覆盖采样四参数（max_tokens/system_prompt 不动），值对齐 server/params/presets.py
type PresetKey = 'precise' | 'balanced' | 'creative'
type SamplingValues = Pick<
  InferenceParams,
  'temperature' | 'top_p' | 'top_k' | 'repetition_penalty'
>

const PRESETS: Record<PresetKey, { label: string; values: SamplingValues }> = {
  precise: { label: '精确', values: { temperature: 0.1, top_p: 0.5, top_k: 10, repetition_penalty: 1.05 } },
  balanced: { label: '平衡', values: { temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05 } },
  creative: { label: '创意', values: { temperature: 1.0, top_p: 0.9, top_k: 40, repetition_penalty: 1.05 } },
}

const PRESET_LABELS: Record<string, string> = {
  precise: '精确',
  balanced: '平衡',
  creative: '创意',
  custom: '自定义',
}

const props = defineProps<{ visible: boolean }>()
const emit = defineEmits<{ 'update:visible': [value: boolean] }>()

const settingsStore = useSettingsStore()
// 移动端抽屉全屏（M4）
const isMobile = useIsMobile()

const drawerVisible = computed({
  get: () => props.visible,
  set: (value: boolean) => emit('update:visible', value),
})

// 抽屉内编辑副本，保存成功才写回 store；关闭即丢弃
const form = reactive<InferenceParams>({ ...settingsStore.params })
const saving = ref(false)

// 打开抽屉时从后端拉最新参数初始化（对齐「加载时 GET」要求）
watch(
  () => props.visible,
  async (visible) => {
    if (!visible) return
    try {
      await settingsStore.load()
      Object.assign(form, settingsStore.params)
    } catch (e) {
      ElMessage.error((e as ApiError).message ?? '加载参数失败')
    }
  },
)

// 当前表单值命中哪个预设；不命中即 custom（与后端 _match_preset 逻辑一致）
const currentPreset = computed(() => {
  for (const [key, preset] of Object.entries(PRESETS)) {
    if (
      form.temperature === preset.values.temperature &&
      form.top_p === preset.values.top_p &&
      form.top_k === preset.values.top_k &&
      form.repetition_penalty === preset.values.repetition_penalty
    ) {
      return key
    }
  }
  return 'custom'
})

function applyPreset(key: string) {
  const preset = PRESETS[key as PresetKey]
  if (preset) Object.assign(form, preset.values)
}

// 校验规则与后端 _validate 一致，先拦一道；后端 400 时把 message 提示给用户
function validate(): string | null {
  if (form.temperature < 0) return 'temperature 必须 ≥ 0'
  if (!(form.top_p > 0 && form.top_p <= 1)) return 'top_p 必须在 (0, 1] 区间'
  if (!Number.isInteger(form.top_k) || form.top_k < 0) return 'top_k 必须是非负整数'
  if (form.repetition_penalty <= 0) return 'repetition_penalty 必须 > 0'
  if (!Number.isInteger(form.max_tokens) || form.max_tokens < 1) return 'max_tokens 必须是正整数'
  return null
}

async function onSave() {
  const error = validate()
  if (error) {
    ElMessage.error(error)
    return
  }
  saving.value = true
  try {
    // 只发显式字段，不发 preset：值命中预设时服务端会自动标出预设名（preset=custom 传过去会被拒）
    await settingsStore.save({
      temperature: form.temperature,
      top_p: form.top_p,
      top_k: form.top_k,
      repetition_penalty: form.repetition_penalty,
      max_tokens: form.max_tokens,
      system_prompt: form.system_prompt,
    })
    Object.assign(form, settingsStore.params)
    ElMessage.success('参数已保存')
  } catch (e) {
    ElMessage.error((e as ApiError).message ?? '保存失败')
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <el-drawer v-model="drawerVisible" title="推理参数" :size="isMobile ? '100%' : '420px'">
    <div class="params-form">
      <div class="preset-row">
        <el-button
          v-for="(preset, key) in PRESETS"
          :key="key"
          size="small"
          :type="currentPreset === key ? 'primary' : 'default'"
          @click="applyPreset(key)"
        >
          {{ preset.label }}
        </el-button>
        <el-tag size="small" disable-transitions>
          当前：{{ PRESET_LABELS[currentPreset] ?? currentPreset }}
        </el-tag>
      </div>

      <div class="field">
        <span class="field-label">temperature（≥ 0）</span>
        <el-slider v-model="form.temperature" :min="0" :max="2" :step="0.05" show-input />
      </div>
      <div class="field">
        <span class="field-label">top_p（0, 1]</span>
        <el-slider v-model="form.top_p" :min="0" :max="1" :step="0.05" show-input />
      </div>
      <div class="field">
        <span class="field-label">top_k（非负整数）</span>
        <el-slider v-model="form.top_k" :min="0" :max="100" :step="1" show-input />
      </div>
      <div class="field">
        <span class="field-label">repetition_penalty（&gt; 0）</span>
        <el-slider v-model="form.repetition_penalty" :min="0" :max="2" :step="0.05" show-input />
      </div>
      <div class="field">
        <span class="field-label">max_tokens（正整数）</span>
        <el-slider v-model="form.max_tokens" :min="1" :max="8192" :step="1" show-input />
      </div>
      <div class="field">
        <span class="field-label">system prompt</span>
        <el-input
          v-model="form.system_prompt"
          type="textarea"
          :rows="4"
          placeholder="留空则不使用 system prompt"
        />
      </div>

      <el-button type="primary" :loading="saving" @click="onSave">保存</el-button>
    </div>
  </el-drawer>
</template>

<style scoped>
.params-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.preset-row {
  display: flex;
  align-items: center;
  gap: 8px;
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
