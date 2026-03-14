export type TestingStrategy = 'random' | 'progressive'

// 运行参数（与 SSQ 对齐）
export interface TestingRunParams {
  methods: string[]
  strategy: TestingStrategy
  target_prize: string
  periods_start: number
  periods_end: number
  count_start: number
  count_end: number
  max_tests: number
  parallel: boolean
  workers: number
  seed?: number | null
  target_issue?: string | null
  progressive_step?: number
  timeout_seconds?: number
  retries?: number
}

// 方法信息（带分类）
export interface MethodInfo {
  id: string
  name: string
  category: string
}

// 测试选项（后端返回）
export interface TestingOptions {
  methods: MethodInfo[]
  target_prizes: string[]
}

// SSE 事件类型
export interface SseLogEvent {
  message: string
  level: string
}

export interface SseProgressEvent {
  method: string
  periods: number
  strategy?: string
  attempt?: number
  total?: number
  range_start?: number
  range_end?: number
}

export interface SseWinningEvent {
  method: string
  periods: number
  count: number
  prize_level: string
  predicted_fronts: number[]
  predicted_backs: number[]
  winning_fronts: number[]
  winning_backs: number[]
  issue: string
  date: string
}

export interface SseCompleteEvent {
  success: boolean
  message?: string
  session_id?: string
  strategy?: string
  target_prize?: string
  tested_methods?: string[]
  successful_methods?: string[]
  stats?: {
    total_tests: number
    winning_tests: number
    winning_rate: number
    method_stats: Record<string, any>
    prize_stats: Record<string, number>
  }
  report_files?: {
    json: string
    text: string
  }
  time?: string
}

// 运行结果（前端组装）
export interface TestingRunResult {
  session_id: string
  strategy: string
  target_prize: string
  tested_methods: string[]
  successful_methods: string[]
  stats: {
    total_tests: number
    winning_tests: number
    winning_rate: number
    method_stats: Record<string, any>
    prize_stats: Record<string, number>
  }
  report_files: {
    json: string
    text: string
  }
  time: string
}

// 兼容旧类型（避免其他文件报错）
export type TestingRequest = TestingRunParams
export type TestingSummary = TestingRunResult
export interface TestingEvent {
  type: 'log' | 'progress' | 'result' | 'winning' | 'complete' | 'error'
  data: Record<string, any>
}
