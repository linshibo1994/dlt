import request from './index'
import type { ApiResponse, SystemInfo, HealthStatus } from '@/types'

// Get system info
export const getSystemInfo = () => {
  return request.get<any, ApiResponse<SystemInfo>>('/system/info')
}

// Health check
export const checkHealth = () => {
  return request.get<any, ApiResponse<HealthStatus>>('/system/health')
}
