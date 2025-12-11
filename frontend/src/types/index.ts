// Common API Response Wrapper
export interface ApiResponse<T = any> {
  success: boolean
  data: T
  message: string
}

// Lottery Ball Type
export type BallType = 'front' | 'back'

// Prediction Method Categories
export type MethodCategory =
  | 'basic'      // Basic Statistics
  | 'markov'     // Markov Chain
  | 'deep'       // Deep Learning
  | 'ensemble'   // Ensemble Learning
  | 'smart'      // Smart Prediction
  | 'compound'   // Compound Betting

// Acceleration Modes
export type AccelerationMode = 'auto' | 'cpu' | 'cpu_multi' | 'gpu' | 'gpu_cuda'

// Prediction Request Body (完整参数定义，与后端 schema 对齐)
export interface PredictionRequest {
  method: string
  count: number
  periods: number
  front_count?: number
  back_count?: number
  compound_mode?: boolean
  // 胆拖相关参数
  front_dan?: number
  back_dan?: number
  front_tuo?: number
  back_tuo?: number
  // 分析和预测参数
  analysis_periods?: number
  predict_periods?: number
  strategy?: string
  integration_level?: string
  integration_type?: string
  // 投注限制
  max_cost?: number
  min_confidence?: number
  // 硬件加速参数
  acceleration?: string
  cpu_threads?: number
  gpu_device?: number
  gpu_memory_limit?: number
  mixed_precision?: boolean
  batch_size_multiplier?: number
  benchmark_hardware?: boolean
  fallback_enabled?: boolean
  // 训练参数
  auto_epochs?: boolean
  min_epochs?: number
  max_epochs?: number
  performance_mode?: string
  training_intensity?: number
  [key: string]: any // For other dynamic fields
}

// Compare Request Body
export interface CompareRequest {
  target_issue: string
  method?: string
  periods?: number
  times?: number
  random_periods?: boolean
  min_periods?: number
  max_periods?: number | null
  export_excel?: boolean
  show_progress?: boolean
}

// Prize Grade (奖项信息，兼容多种后端返回格式)
export interface PrizeGrade {
  // 新格式 (sporttery.cn API)
  prizeLevelName?: string
  prizeNum?: number
  prizeAmount?: string
  // 旧格式
  level?: number
  grade?: number
  count?: number
  prize_count?: number
  amount?: number
  prize_amount?: number
}

// Lottery Result (Data)
export interface LotteryResult {
  issue: string
  date: string
  front_balls: number[]
  back_balls: number[]
  sales?: number
  pool_money?: number
  prize_grades?: PrizeGrade[]
}

// Data Statistics (与后端 DataStatsResponse 对齐)
export interface DataStats {
  total_periods: number
  date_range: {
    start: string
    end: string
  }
  latest_issue: string | null
  cache?: {
    cache_dir: string
    total: number
    size_mb: number
  }
}

// Algorithm/Method Info
export interface AlgorithmInfo {
  id: string
  name: string
  description: string
  category: string
  support_compound: boolean
}

// Prediction Result Item (method 和 confidence 可能为空)
export interface PredictionResult {
  front_balls: number[]
  back_balls: number[]
  confidence?: number
  method?: string
  details?: Record<string, any>
}

// Compound Prediction Result (与后端 CompoundResultModel 对齐)
export interface CompoundResultModel {
  front_balls: number[]
  back_balls: number[]
  front_count?: number
  back_count?: number
  total_combinations?: number
  total_cost?: number
  confidence?: number
  method?: string
  analysis_periods?: number
  timestamp?: string
  details?: Record<string, any>
  combinations?: Array<{
    front_balls: number[]
    back_balls: number[]
  }>
}

// Prediction Response Data (后端返回的完整结构)
export interface PredictionResponseData {
  mode: string // 'traditional' | 'enhanced' | 'deep_learning'
  predictions: PredictionResult[]
  compound?: CompoundResultModel | null
}

// Compare Result Item
export interface CompareResult {
  index: number
  predicted_front: number[]
  predicted_back: number[]
  actual_front: number[]
  actual_back: number[]
  prize_level: string
  front_hits: number
  back_hits: number
}

// System Info
export interface SystemInfo {
  platform: string
  processor: string
  python_version: string
  cpu_usage: number
  memory_total: number
  memory_available: number
  memory_percent: number
  disk_total: number
  disk_free: number
  disk_percent: number
}

// Health Status
export interface HealthStatus {
  status: 'healthy' | 'unhealthy'
  timestamp: string
  services: Record<string, 'up' | 'down'>
}