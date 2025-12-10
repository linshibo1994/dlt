<template>
  <div class="glass-card" :class="{ 'glow': glow, 'hoverable': hoverable }">
    <div v-if="title || $slots.headerExtra" class="card-header">
      <h3 v-if="title" class="card-title">{{ title }}</h3>
      <div class="header-extra">
        <slot name="headerExtra" />
      </div>
    </div>
    <div class="card-content" :class="{ 'no-padding': noPadding }">
      <slot />
    </div>
    <div v-if="$slots.footer" class="card-footer">
      <slot name="footer" />
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title?: string
  glow?: boolean
  hoverable?: boolean
  noPadding?: boolean
}>()
</script>

<style scoped>
.glass-card {
  background: var(--bg-card);
  backdrop-filter: blur(var(--blur-glass));
  -webkit-backdrop-filter: blur(var(--blur-glass));
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-glow);
  overflow: hidden;
  transition: all var(--transition-normal);
}

.glass-card.hoverable:hover {
  background: var(--bg-card-hover);
  border-color: rgba(0, 212, 255, 0.4);
  transform: translateY(-4px);
  box-shadow: var(--shadow-glass);
}

.glass-card.glow {
  box-shadow: 0 0 40px rgba(0, 212, 255, 0.2);
  animation: glow 2s ease-in-out infinite alternate;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border-default);
}

.card-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.header-extra {
  display: flex;
  align-items: center;
  gap: 8px;
}

.card-content {
  padding: 24px;
}

.card-content.no-padding {
  padding: 0;
}

.card-footer {
  padding: 16px 24px;
  border-top: 1px solid var(--border-default);
  background: rgba(0, 0, 0, 0.2);
}

@keyframes glow {
  0% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.4); }
  100% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.8); }
}
</style>
