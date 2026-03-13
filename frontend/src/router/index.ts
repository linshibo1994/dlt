import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Dashboard',
    component: () => import('@/views/Dashboard.vue'),
    meta: { title: '系统概览', icon: 'HomeOutline' }
  },
  {
    path: '/predict',
    name: 'Prediction',
    component: () => import('@/views/Prediction.vue'),
    meta: { title: '号码预测', icon: 'DiceOutline' }
  },
  {
    path: '/analysis',
    name: 'Analysis',
    component: () => import('@/views/Analysis.vue'),
    meta: { title: '数据分析', icon: 'BarChartOutline' }
  },
  {
    path: '/compare',
    name: 'Compare',
    component: () => import('@/views/Compare.vue'),
    meta: { title: '批量对比', icon: 'GitCompareOutline' }
  },
  {
    path: '/testing',
    name: 'Testing',
    component: () => import('@/views/Testing.vue'),
    meta: { title: '测试系统', icon: 'FlaskOutline' }
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('@/views/Settings.vue'),
    meta: { title: '系统设置', icon: 'SettingsOutline' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
