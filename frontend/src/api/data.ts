import request from './index'
import type { ApiResponse, LotteryResult, DataStats } from '@/types'

// Get latest lottery data
export const getLatestData = () => {
  return request.get<any, ApiResponse<LotteryResult>>('/data/latest')
}

// Get history data with pagination
export const getHistoryData = (page: number = 1, pageSize: number = 50) => {
  return request.get<any, ApiResponse<{ items: LotteryResult[]; total: number; page: number; page_size: number }>>('/data/history', {
    params: {
      page,
      page_size: pageSize
    }
  })
}

// Get data statistics
export const getDataStats = () => {
  return request.get<any, ApiResponse<DataStats>>('/data/stats')
}