import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export interface AppSettings {
  // 基础设置
  apiBaseUrl: string
  defaultPeriods: number
  defaultCount: number
  autoSave: boolean
  // 界面设置
  enableAnimations: boolean
  ballGlow: boolean
  soundEnabled: boolean
  compactMode: boolean
  // 算法设置
  defaultAlgorithms: string[]
  gpuAcceleration: boolean
  parallelCompute: boolean
  confidenceThreshold: number
  // 数据设置
  autoUpdate: boolean
  updateInterval: string
  cacheStrategy: string
  // 兼容旧字段
  theme: 'dark' | 'light'
  language: 'zh-CN' | 'en-US'
}

const defaultSettings: AppSettings = {
  apiBaseUrl: 'http://localhost:8000',
  defaultPeriods: 200,
  defaultCount: 5,
  autoSave: true,
  enableAnimations: true,
  ballGlow: true,
  soundEnabled: false,
  compactMode: false,
  defaultAlgorithms: ['ensemble', 'lstm'],
  gpuAcceleration: false,
  parallelCompute: true,
  confidenceThreshold: 60,
  autoUpdate: true,
  updateInterval: '1h',
  cacheStrategy: '24h',
  theme: 'dark',
  language: 'zh-CN'
}

export const useSettingsStore = defineStore('settings', () => {
  // 从localStorage加载设置，带类型验证
  const loadSettings = (): AppSettings => {
    try {
      const saved = localStorage.getItem('app-settings')
      if (saved) {
        const parsed = JSON.parse(saved)
        if (typeof parsed === 'object' && parsed !== null) {
          return {
            ...defaultSettings,
            ...parsed,
            // 确保关键字段类型正确
            defaultPeriods: typeof parsed.defaultPeriods === 'number' ? parsed.defaultPeriods : defaultSettings.defaultPeriods,
            defaultCount: typeof parsed.defaultCount === 'number' ? parsed.defaultCount : defaultSettings.defaultCount,
            enableAnimations: typeof parsed.enableAnimations === 'boolean' ? parsed.enableAnimations : defaultSettings.enableAnimations,
            defaultAlgorithms: Array.isArray(parsed.defaultAlgorithms) ? parsed.defaultAlgorithms : defaultSettings.defaultAlgorithms
          }
        }
      }
    } catch (error) {
      console.error('加载设置失败:', error)
    }
    return defaultSettings
  }

  // 状态
  const settings = ref<AppSettings>(loadSettings())

  // 保存设置
  const saveSettings = () => {
    try {
      localStorage.setItem('app-settings', JSON.stringify(settings.value))
    } catch (error) {
      console.error('保存设置失败:', error)
    }
  }

  // 更新设置
  const updateSettings = (newSettings: Partial<AppSettings>) => {
    settings.value = { ...settings.value, ...newSettings }
    saveSettings()
  }

  // 重置设置
  const resetSettings = () => {
    settings.value = { ...defaultSettings }
    saveSettings()
  }

  // 清除应用缓存（不包括用户设置）
  const clearCache = () => {
    const keysToRemove = ['prediction-cache', 'analysis-cache', 'lottery-data-cache']
    keysToRemove.forEach(key => localStorage.removeItem(key))
  }

  // 监听设置变化并保存
  watch(settings, saveSettings, { deep: true })

  return {
    settings,
    updateSettings,
    resetSettings,
    clearCache
  }
})
