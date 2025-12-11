import request from './index'
import type { ApiResponse, CompareRequest, CompareResult } from '@/types'

// Compare response data type (matches backend CompareResponseData)
interface CompareResponseData {
  summary: any
  records: CompareResult[]
}

// Execute batch comparison
export const executeBatchComparison = (data: CompareRequest) => {
  return request.post<any, ApiResponse<CompareResponseData>>('/compare', data)
}
