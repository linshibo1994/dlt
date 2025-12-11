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

  return {
    status,
    latestResult,
    isLoading,
    isUpdating,
    lastUpdateResult,
    fetchStatus,
    fetchLatestResult,
    updateData
  }
})