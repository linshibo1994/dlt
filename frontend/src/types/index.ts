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

// Prediction Request Body
export interface PredictionRequest {
  method: string
  count: number
  periods: number
  front_count: number
  back_count: number
  compound_mode: boolean
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

// Lottery Result (Data)
export interface LotteryResult {
  issue: string
  date: string
  front_balls: number[]
  back_balls: number[]
  sales?: number
  pool_money?: number
  prize_grades?: any[]
}

// Data Statistics
export interface DataStats {
  total_periods: number
  start_date: string
  end_date: string
  last_updated: string
}

// Algorithm/Method Info
export interface AlgorithmInfo {
  id: string
  name: string
  description: string
  category: string
  support_compound: boolean
}

// Prediction Result Item
export interface PredictionResult {
  front_balls: number[]
  back_balls: number[]
  confidence: number
  method: string
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