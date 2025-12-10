<template>
  <div
    class="lottery-ball"
    :class="[type, size, { 'animate': animate, 'glow': glow }]"
    :style="customStyle"
  >
    <span class="ball-number">{{ formattedNumber }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { BallType } from '@/types'

const props = withDefaults(defineProps<{
  number: number
  type?: BallType
  size?: 'sm' | 'md' | 'lg' | 'xl'
  animate?: boolean
  glow?: boolean
  delay?: number
}>(), {
  type: 'front',
  size: 'md',
  animate: false,
  glow: false,
  delay: 0
})

const formattedNumber = computed(() => {
  return props.number.toString().padStart(2, '0')
})

const customStyle = computed(() => {
  if (props.delay > 0) {
    return { animationDelay: `${props.delay}ms` }
  }
  return {}
})
</script>

<style scoped>
.lottery-ball {
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  font-weight: bold;
  transition: all 0.3s ease;
  user-select: none;
}

/* 尺寸 */
.lottery-ball.sm {
  width: 32px;
  height: 32px;
  font-size: 12px;
}

.lottery-ball.md {
  width: 48px;
  height: 48px;
  font-size: 18px;
}

.lottery-ball.lg {
  width: 64px;
  height: 64px;
  font-size: 24px;
}

.lottery-ball.xl {
  width: 80px;
  height: 80px;
  font-size: 32px;
}

/* 前区球 - 霓虹蓝 */
.lottery-ball.front {
  background: linear-gradient(135deg, var(--ball-front-primary) 0%, var(--ball-front-secondary) 100%);
  box-shadow: var(--ball-front-glow), inset 0 -3px 10px rgba(0, 0, 0, 0.2);
  color: var(--bg-primary);
}

/* 后区球 - 霓虹紫 */
.lottery-ball.back {
  background: linear-gradient(135deg, var(--ball-back-primary) 0%, var(--ball-back-secondary) 100%);
  box-shadow: var(--ball-back-glow), inset 0 -3px 10px rgba(0, 0, 0, 0.2);
  color: var(--bg-primary);
}

/* 入场动画 */
.lottery-ball.animate {
  animation: ballPop 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) both;
}

@keyframes ballPop {
  0% {
    transform: scale(0) rotate(-180deg);
    opacity: 0;
  }
  50% {
    transform: scale(1.2) rotate(0deg);
  }
  100% {
    transform: scale(1) rotate(0deg);
    opacity: 1;
  }
}

/* 发光效果 */
.lottery-ball.glow.front {
  animation: glowFront 2s ease-in-out infinite alternate;
}

.lottery-ball.glow.back {
  animation: glowBack 2s ease-in-out infinite alternate;
}

@keyframes glowFront {
  0% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.6); }
  100% { box-shadow: 0 0 40px rgba(0, 212, 255, 1); }
}

@keyframes glowBack {
  0% { box-shadow: 0 0 20px rgba(255, 0, 255, 0.6); }
  100% { box-shadow: 0 0 40px rgba(255, 0, 255, 1); }
}

/* 悬停效果 */
.lottery-ball:hover {
  transform: scale(1.1);
}

.lottery-ball.front:hover {
  box-shadow: 0 0 30px rgba(0, 212, 255, 0.8);
}

.lottery-ball.back:hover {
  box-shadow: 0 0 30px rgba(255, 0, 255, 0.8);
}

.ball-number {
  position: relative;
  z-index: 1;
}
</style>
