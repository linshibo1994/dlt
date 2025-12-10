<template>
  <div class="progress-ring-container">
    <svg class="progress-ring" :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`">
      <defs>
        <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="var(--neon-blue)" />
          <stop offset="100%" stop-color="var(--neon-purple)" />
        </linearGradient>
      </defs>
      <circle
        class="progress-bg"
        :cx="center"
        :cy="center"
        :r="radius"
        fill="none"
        :stroke-width="strokeWidth"
      />
      <circle
        class="progress-bar"
        :cx="center"
        :cy="center"
        :r="radius"
        fill="none"
        :stroke-width="strokeWidth"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="dashOffset"
      />
    </svg>
    <div class="progress-content">
      <span class="progress-value">{{ Math.round(value) }}</span>
      <span class="progress-unit">%</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  value: number
  size?: number
  strokeWidth?: number
}>(), {
  size: 120,
  strokeWidth: 8
})

const center = computed(() => props.size / 2)
const radius = computed(() => (props.size - props.strokeWidth) / 2)
const circumference = computed(() => 2 * Math.PI * radius.value)
const dashOffset = computed(() => {
  const progress = Math.min(Math.max(props.value, 0), 100)
  return circumference.value * (1 - progress / 100)
})
</script>

<style scoped>
.progress-ring-container {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.progress-ring {
  transform: rotate(-90deg);
}

.progress-bg {
  stroke: rgba(0, 212, 255, 0.1);
}

.progress-bar {
  stroke: url(#progressGradient);
  stroke-linecap: round;
  transition: stroke-dashoffset 0.5s ease;
}

.progress-content {
  position: absolute;
  display: flex;
  align-items: baseline;
  justify-content: center;
}

.progress-value {
  font-size: 24px;
  font-weight: bold;
  color: var(--neon-blue);
  text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
}

.progress-unit {
  font-size: 14px;
  color: var(--text-secondary);
  margin-left: 2px;
}
</style>
