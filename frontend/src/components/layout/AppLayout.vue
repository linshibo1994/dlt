<template>
  <div class="app-layout">
    <aside class="sidebar" :class="{ 'collapsed': sidebarCollapsed }">
      <div class="logo">
        <div class="logo-icon">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
            <circle cx="12" cy="12" r="4" fill="currentColor"/>
          </svg>
        </div>
        <span v-show="!sidebarCollapsed" class="logo-text">大乐透预测</span>
      </div>

      <nav class="nav-menu">
        <router-link
          v-for="item in menuItems"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ 'active': isActive(item.path) }"
        >
          <component :is="item.icon" class="nav-icon" />
          <span v-show="!sidebarCollapsed" class="nav-label">{{ item.label }}</span>
        </router-link>
      </nav>

      <button class="collapse-btn" @click="toggleSidebar">
        <ChevronBackOutline v-if="!sidebarCollapsed" class="collapse-icon" />
        <ChevronForwardOutline v-else class="collapse-icon" />
      </button>
    </aside>

    <main class="main-content">
      <header class="header">
        <div class="header-left">
          <h1 class="page-title">{{ currentPageTitle }}</h1>
        </div>
        <div class="header-right">
          <div class="status-indicator">
            <span class="status-dot"></span>
            <span class="status-text">系统正常</span>
          </div>
        </div>
      </header>

      <div class="content-wrapper">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'
import {
  HomeOutline,
  DiceOutline,
  BarChartOutline,
  TimeOutline,
  GitCompareOutline,
  FlaskOutline,
  SettingsOutline,
  ChevronBackOutline,
  ChevronForwardOutline
} from '@vicons/ionicons5'

const route = useRoute()
const sidebarCollapsed = ref(false)

const menuItems = [
  { path: '/', label: '系统概览', icon: HomeOutline },
  { path: '/predict', label: '号码预测', icon: DiceOutline },
  { path: '/analysis', label: '数据分析', icon: BarChartOutline },
  { path: '/history', label: '历史数据', icon: TimeOutline },
  { path: '/compare', label: '批量对比', icon: GitCompareOutline },
  { path: '/testing', label: '测试系统', icon: FlaskOutline },
  { path: '/settings', label: '系统设置', icon: SettingsOutline }
]

const currentPageTitle = computed(() => {
  const current = menuItems.find(item => item.path === route.path)
  return current?.label || '大乐透预测系统'
})

const isActive = (path: string) => {
  return route.path === path
}

const toggleSidebar = () => {
  sidebarCollapsed.value = !sidebarCollapsed.value
}
</script>

<style scoped>
.app-layout {
  display: flex;
  min-height: 100vh;
}

/* 侧边栏 */
.sidebar {
  width: 240px;
  background: var(--bg-card);
  backdrop-filter: blur(var(--blur-glass));
  border-right: 1px solid var(--border-default);
  display: flex;
  flex-direction: column;
  transition: width var(--transition-normal);
  position: fixed;
  height: 100vh;
  z-index: 100;
}

.sidebar.collapsed {
  width: 72px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 24px 20px;
  border-bottom: 1px solid var(--border-default);
}

.logo-icon {
  width: 32px;
  height: 32px;
  color: var(--neon-blue);
  flex-shrink: 0;
}

.logo-text {
  font-size: 18px;
  font-weight: bold;
  color: var(--text-primary);
  white-space: nowrap;
}

.nav-menu {
  flex: 1;
  padding: 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  transition: all var(--transition-fast);
  text-decoration: none;
}

.nav-item:hover {
  background: rgba(0, 212, 255, 0.1);
  color: var(--text-primary);
}

.nav-item.active {
  background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(255, 0, 255, 0.1));
  color: var(--neon-blue);
  border: 1px solid var(--border-glow);
}

.nav-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.nav-label {
  white-space: nowrap;
}

.collapse-btn {
  margin: 16px 12px;
  padding: 12px;
  background: transparent;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition-fast);
  display: flex;
  align-items: center;
  justify-content: center;
}

.collapse-btn:hover {
  background: rgba(0, 212, 255, 0.1);
  border-color: var(--border-glow);
  color: var(--neon-blue);
}

.collapse-icon {
  width: 20px;
  height: 20px;
}

/* 主内容区 */
.main-content {
  flex: 1;
  margin-left: 240px;
  transition: margin-left var(--transition-normal);
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.sidebar.collapsed + .main-content {
  margin-left: 72px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 32px;
  background: var(--bg-card);
  backdrop-filter: blur(var(--blur-glass));
  border-bottom: 1px solid var(--border-default);
  position: sticky;
  top: 0;
  z-index: 50;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: rgba(0, 255, 136, 0.1);
  border: 1px solid rgba(0, 255, 136, 0.3);
  border-radius: var(--radius-full);
}

.status-dot {
  width: 8px;
  height: 8px;
  background: var(--neon-green);
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}

.status-text {
  font-size: 12px;
  color: var(--neon-green);
}

.content-wrapper {
  flex: 1;
  padding: 32px;
  overflow-y: auto;
}

/* 页面切换动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.fade-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.fade-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
