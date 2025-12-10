<template>
  <div class="loading-spinner" :class="size">
    <div class="spinner">
      <div class="spinner-ring" v-for="i in 3" :key="i"></div>
    </div>
    <p v-if="message" class="loading-text">{{ message }}</p>
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  message?: string
  size?: 'sm' | 'md' | 'lg'
}>(), {
  size: 'md'
})
</script>

<style scoped>
.loading-spinner {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
}

.spinner {
  position: relative;
}

/* 尺寸 */
.loading-spinner.sm .spinner {
  width: 40px;
  height: 40px;
}

.loading-spinner.md .spinner {
  width: 60px;
  height: 60px;
}

.loading-spinner.lg .spinner {
  width: 80px;
  height: 80px;
}

.spinner-ring {
  position: absolute;
  width: 100%;
  height: 100%;
  border: 3px solid transparent;
  border-radius: 50%;
  animation: spin 1.5s linear infinite;
}

.spinner-ring:nth-child(1) {
  border-top-color: var(--neon-blue);
  animation-delay: 0s;
}

.spinner-ring:nth-child(2) {
  border-right-color: var(--neon-purple);
  animation-delay: 0.2s;
}

.spinner-ring:nth-child(3) {
  border-bottom-color: var(--neon-green);
  animation-delay: 0.4s;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-text {
  color: var(--text-secondary);
  font-size: 14px;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
