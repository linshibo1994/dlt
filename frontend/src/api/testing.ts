import request from './index'
import type { ApiResponse } from '@/types'
import type { TestingOptions, TestingRunParams } from '@/types/testing'

// 获取测试选项（方法列表+奖级列表）
export const getTestingOptions = () => {
  return request.get<any, ApiResponse<TestingOptions>>('/testing/options')
}

// 创建 SSE 流式测试连接
export const createTestingStream = (data: TestingRunParams): EventSource => {
  const params = new URLSearchParams()
  params.set('methods', data.methods.join(','))
  params.set('strategy', data.strategy)
  params.set('target_prize', data.target_prize)
  params.set('periods_start', String(data.periods_start))
  params.set('periods_end', String(data.periods_end))
  params.set('count_start', String(data.count_start))
  params.set('count_end', String(data.count_end))
  params.set('max_tests', String(data.max_tests))
  params.set('parallel', String(data.parallel))
  params.set('workers', String(data.workers))
  if (typeof data.seed === 'number') params.set('seed', String(data.seed))
  if (data.target_issue) params.set('target_issue', data.target_issue)
  if (typeof data.progressive_step === 'number') params.set('progressive_step', String(data.progressive_step))
  if (typeof data.timeout_seconds === 'number') params.set('timeout_seconds', String(data.timeout_seconds))
  if (typeof data.retries === 'number') params.set('retries', String(data.retries))
  return new EventSource(`/api/testing/stream?${params.toString()}`)
}

// 同步运行测试（备用）
export const runTesting = (data: TestingRunParams) => {
  return request.post<any, ApiResponse<any>>('/testing/run', data)
}
