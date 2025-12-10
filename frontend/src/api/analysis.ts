import request from './index'
import type { ApiResponse } from '@/types'

export interface AnalysisParams {
  periods: number
}

export interface FrequencyResult {
  front_frequency: Record<string, number>
  back_frequency: Record<string, number>
}

export interface HotColdResult {
  front_hot: number[]
  front_cold: number[]
  back_hot: number[]
  back_cold: number[]
}

export interface TrendResult {
  front_trend: any[]
  back_trend: any[]
}

export interface MissingResult {
  front_missing: Record<string, number>
  back_missing: Record<string, number>
}

// Frequency Analysis
export const getFrequencyAnalysis = (data: AnalysisParams) => {
  return request.post<any, ApiResponse<FrequencyResult>>('/analysis/frequency', data)
}

// Hot-Cold Analysis
export const getHotColdAnalysis = (data: AnalysisParams) => {
  return request.post<any, ApiResponse<HotColdResult>>('/analysis/hot-cold', data)
}

// Trend Analysis
export const getTrendAnalysis = (data: AnalysisParams) => {
  return request.post<any, ApiResponse<TrendResult>>('/analysis/trend', data)
}

// Missing Analysis
export const getMissingAnalysis = (data: AnalysisParams) => {
  return request.post<any, ApiResponse<MissingResult>>('/analysis/missing', data)
}