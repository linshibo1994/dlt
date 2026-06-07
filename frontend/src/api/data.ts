import request from './index'
import type { ApiResponse, LotteryResult, DataStats } from '@/types'

// Data update response type
export interface DataUpdateResult {
  updated_count: number
  total_periods: number
  latest_issue: string | null
  latest_date: string | null
  source?: string | null
  reason?: string | null
  updated_at?: string | null
  skipped?: boolean
  message?: string
}

// Get latest lottery data
export const getLatestData = () => {
  return request.get<any, ApiResponse<LotteryResult>>('/data/latest')
}

// Get history data with pagination
export const getHistoryData = (page: number = 1, pageSize: number = 50) => {
  return request.get<any, ApiResponse<{ records: LotteryResult[]; total: number; page: number; page_size: number }>>('/data/history', {
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

// Update lottery data (crawl from official website)
export const updateLotteryData = () => {
  return request.post<any, ApiResponse<DataUpdateResult>>('/data/update')
}
