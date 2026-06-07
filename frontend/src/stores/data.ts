import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { DataStats, LotteryResult } from '@/types'
import { getDataStats, getLatestData, updateLotteryData, type DataUpdateResult } from '@/api/data'

export const useDataStore = defineStore('data', () => {
  // 状态
  const status = ref<DataStats | null>(null)
  const latestResult = ref<LotteryResult | null>(null)
  const isLoading = ref(false)
  const isUpdating = ref(false)
  const lastUpdateResult = ref<DataUpdateResult | null>(null)
  const hasAutoUpdated = ref(false)
  let autoUpdatePromise: Promise<{ success: boolean; message: string; data?: DataUpdateResult }> | null = null

  // 获取数据状态
  const fetchStatus = async () => {
    try {
      isLoading.value = true
      const response = await getDataStats()
      if (response.success) {
        status.value = response.data
      }
    } catch (error) {
      console.error('获取数据状态失败:', error)
    } finally {
      isLoading.value = false
    }
  }

  // 获取最新开奖结果
  const fetchLatestResult = async () => {
    try {
      const response = await getLatestData()
      if (response.success) {
        latestResult.value = response.data
      }
    } catch (error) {
      console.error('获取最新结果失败:', error)
    }
  }

  // 更新数据（从官网爬取）
  const updateData = async (): Promise<{ success: boolean; message: string; data?: DataUpdateResult }> => {
    try {
      isUpdating.value = true
      const response = await updateLotteryData()
      if (response.success) {
        lastUpdateResult.value = response.data
        // 更新后重新获取最新数据
        await fetchLatestResult()
        await fetchStatus()
        return { success: true, message: response.message, data: response.data }
      }
      return { success: false, message: response.message || '更新失败' }
    } catch (error: any) {
      console.error('更新数据失败:', error)
      return { success: false, message: error.message || '更新数据失败' }
    } finally {
      isUpdating.value = false
    }
  }

  // 页面打开或浏览器刷新时自动增量更新一次
  const autoUpdateLatestData = async (): Promise<{ success: boolean; message: string; data?: DataUpdateResult }> => {
    if (autoUpdatePromise) {
      return autoUpdatePromise
    }

    if (hasAutoUpdated.value) {
      return {
        success: true,
        message: '已完成本次页面自动更新',
        data: lastUpdateResult.value || undefined
      }
    }

    hasAutoUpdated.value = true
    autoUpdatePromise = updateData()
      .then((result) => {
        if (!result.success) {
          hasAutoUpdated.value = false
        }
        return result
      })
      .catch((error) => {
        hasAutoUpdated.value = false
        throw error
      })
      .finally(() => {
        autoUpdatePromise = null
      })

    return autoUpdatePromise
  }

  return {
    status,
    latestResult,
    isLoading,
    isUpdating,
    lastUpdateResult,
    hasAutoUpdated,
    fetchStatus,
    fetchLatestResult,
    updateData,
    autoUpdateLatestData
  }
})
