export type TestingStrategy = 'random' | 'progressive'

export interface TestingRequest {
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

export interface TestingOptions {
  available_methods: string[]
  target_prizes: string[]
}

export interface TestingMethodOutcome {
  method: string
  tests_run: number
  hit_target: boolean
  best_prize_level: number
  best_prize_name: string
  error?: string
}

export interface TestingSummary {
  session_id: string
  test_time: string
  total_tests: number
  winning_tests: number
  winning_rate: number
  execution_time: number
  method_stats: Record<string, any>
  prize_stats: Record<string, number>
  best_methods: Array<{ method: string; score: number; best_prize: string }>
  method_outcomes: TestingMethodOutcome[]
  report_files: {
    json: string
    txt: string
    log: string
  }
  config: {
    methods: string[]
    strategy: TestingStrategy
    target_prize: string
    periods_range: [number, number]
    count_range: [number, number]
    max_tests: number
    parallel: boolean
    workers: number
    progressive_step: number
  }
  target_draw: {
    issue: string
    date: string
    front_balls: string
    back_balls: string
  }
}

export interface TestingEvent {
  type: 'log' | 'progress' | 'result' | 'winning' | 'complete' | 'error'
  data: Record<string, any>
}
