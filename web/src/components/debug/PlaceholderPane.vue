<script setup lang="ts">
// 一期占位面板：所有标签页统一渲染「未实现」（PRD 3.7）
// 二期新增标签页 = 在 debug/ 下写一个组件 + 在 DebugDrawer 的 TABS 配置里加一项
defineProps<{ title: string }>()
</script>

<template>
  <div class="placeholder-pane">
    <el-empty :description="`${title}：未实现`" />
  </div>
</template>

<style scoped>
.placeholder-pane {
  padding: 24px 0;
}
</style>
