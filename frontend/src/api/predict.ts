import request from './index'
import type { ApiResponse, AlgorithmInfo, PredictionRequest, PredictionResponseData } from '@/types'

// List all available algorithms
export const getAlgorithms = () => {
  return request.get<any, ApiResponse<AlgorithmInfo[]>>('/algorithms')
}

// Execute prediction
export const executePrediction = (data: PredictionRequest) => {
  return request.post<any, ApiResponse<PredictionResponseData>>('/predict', data)
}

// Get prediction history
export const getPredictionHistory = (page: number = 1, pageSize: number = 20) => {
  return request.get<any, ApiResponse<{ items: any[]; total: number; page: number; page_size: number }>>('/predict/history', {
    params: {
      page,
      page_size: pageSize
    }
  })
}