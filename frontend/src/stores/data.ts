import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { DataStats, LotteryResult } from '@/types'
import { getDataStats, getLatestData } from '@/api/data'

export const useDataStore = defineStore('data', () => {
  // 状态
  const status = ref<DataStats | null>(null)
  const latestResult = ref<LotteryResult | null>(null)
  const isLoading = ref(false)

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

  return {
    status,
    latestResult,
    isLoading,
    fetchStatus,
    fetchLatestResult
  }
})