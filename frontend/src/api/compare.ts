import request from './index'
import type { ApiResponse, CompareRequest, CompareResult } from '@/types'

// Execute batch comparison
export const executeBatchComparison = (data: CompareRequest) => {
  return request.post<any, ApiResponse<{ results: CompareResult[]; summary: any }>>('/compare', data)
}
