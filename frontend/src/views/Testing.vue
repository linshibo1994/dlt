<template>
  <div class="testing-page">
    <GlassCard>
      <div class="page-content">
        <div class="page-header">
          <h2 class="page-title">测试系统</h2>
          <div class="header-actions">
            <button class="btn-secondary" @click="toggleSelectAll" :disabled="isRunning || allMethodIds.length === 0">
              {{ isAllSelected ? '取消全选' : '全选方法' }}
            </button>
            <button class="btn-danger" v-if="isRunning" @click="stopTesting">停止测试</button>
            <button class="btn-primary" v-else @click="startTesting" :disabled="selectedMethods.length === 0">
              开始测试
            </button>
          </div>
        </div>

        <div class="config-grid">
          <div class="config-item">
            <label>目标奖级</label>
            <select v-model="params.target_prize" :disabled="isRunning">
              <option v-for="p in prizeOptions" :key="p" :value="p">{{ p }}</option>
            </select>
          </div>
          <div class="config-item">
            <label>测试策略</label>
            <select v-model="params.strategy" :disabled="isRunning">
              <option value="progressive">渐进测试</option>
              <option value="random">随机测试</option>
            </select>
          </div>
          <div class="config-item">
            <label>每方法最大测试次数</label>
            <input type="number" v-model.number="params.max_tests" min="0" max="5000" :disabled="isRunning" />
          </div>
          <div class="config-item checkbox-item">
            <label>
              <input type="checkbox" v-model="params.parallel" :disabled="isRunning" />
              并行测试
            </label>
          </div>
          <div class="config-item">
            <label>并行线程数</label>
            <input type="number" v-model.number="params.workers" min="1" max="64" :disabled="isRunning || !params.parallel" />
          </div>
          <div class="config-item">
            <label>期数范围</label>
            <div class="range-inputs">
              <input type="number" v-model.number="params.periods_start" min="1" max="20000" :disabled="isRunning" />
              <span>~</span>
              <input type="number" v-model.number="params.periods_end" min="1" max="20000" :disabled="isRunning" />
            </div>
          </div>
          <div class="config-item">
            <label>注数范围</label>
            <div class="range-inputs">
              <input type="number" v-model.number="params.count_start" min="1" max="100" :disabled="isRunning" />
              <span>~</span>
              <input type="number" v-model.number="params.count_end" min="1" max="100" :disabled="isRunning" />
            </div>
          </div>
        </div>

        <div class="method-section">
          <div class="section-header-row">
            <h3 class="section-title">测试方法</h3>
            <span class="method-count">已选 {{ selectedMethods.length }} / {{ allMethodIds.length }}</span>
          </div>

          <div class="method-groups" v-if="Object.keys(methodGroups).length > 0">
            <div v-for="(methods, category) in methodGroups" :key="category" class="method-group">
              <div class="group-title">{{ String(category) }}</div>
              <div class="method-list">
                <label v-for="method in methods" :key="method.id" class="method-item">
                  <input
                    type="checkbox"
                    :value="method.id"
                    v-model="selectedMethods"
                    :disabled="isRunning"
                  />
                  <span>{{ method.name }}</span>
                </label>
              </div>
            </div>
          </div>

          <div v-else class="empty-tip">
            {{ loadingMethods ? '正在加载方法列表...' : '未获取到可测试方法' }}
          </div>
        </div>
      </div>
    </GlassCard>

    <!-- 实时进度指示 -->
    <div v-if="isRunning || currentProgress" class="progress-banner">
      <div class="progress-indicator">
        <span class="spinner"></span>
        <span v-if="currentProgress">
          正在测试：<strong>{{ methodLabel(currentProgress.method) }}</strong>
          &nbsp;|&nbsp;分析期数：<strong>{{ currentProgress.periods }}</strong>
          <template v-if="currentProgress.attempt">
            &nbsp;|&nbsp;第 <strong>{{ currentProgress.attempt }}</strong> / {{ currentProgress.total }} 次
          </template>
          <template v-else-if="currentProgress.range_start">
            &nbsp;|&nbsp;范围 {{ currentProgress.range_start }} ~ {{ currentProgress.range_end }}
          </template>
        </span>
        <span v-else>测试任务准备中...</span>
      </div>
      <div class="progress-stats">
        累计测试：{{ liveStats.total }} &nbsp;|&nbsp;
        中奖：<span class="win-count">{{ liveStats.winning }}</span>
      </div>
    </div>

    <!-- 中奖结果实时展示 -->
    <div v-if="winningRecords.length > 0" class="winning-section">
      <div class="winning-header">
        <h3 class="winning-title">中奖记录</h3>
        <span class="winning-badge">{{ winningRecords.length }} 条</span>
      </div>
      <div class="winning-list">
        <div
          v-for="(win, index) in winningRecords"
          :key="index"
          class="winning-card"
          :class="prizeCls(win.prize_level)"
        >
          <div class="win-card-header">
            <span class="prize-tag">{{ win.prize_level }}</span>
            <span class="win-method">{{ methodLabel(win.method) }}</span>
            <span class="win-time">{{ formatTime(win.timestamp) }}</span>
          </div>
          <div class="win-card-body">
            <div class="win-row">
              <span class="win-label">分析期数</span>
              <span class="win-value">{{ win.periods }} 期</span>
            </div>
            <div class="win-row">
              <span class="win-label">预测号码</span>
              <div class="balls-row">
                <span
                  v-for="n in win.predicted_fronts"
                  :key="'pf-' + n"
                  class="ball front-ball"
                  :class="{ 'ball-hit': win.winning_fronts.includes(n) }"
                >{{ String(n).padStart(2, '0') }}</span>
                <span class="ball-sep">+</span>
                <span
                  v-for="n in win.predicted_backs"
                  :key="'pb-' + n"
                  class="ball back-ball"
                  :class="{ 'ball-hit': win.winning_backs.includes(n) }"
                >{{ String(n).padStart(2, '0') }}</span>
              </div>
            </div>
            <div class="win-row">
              <span class="win-label">开奖号码</span>
              <div class="balls-row">
                <span
                  v-for="n in win.winning_fronts"
                  :key="'wf-' + n"
                  class="ball front-ball"
                >{{ String(n).padStart(2, '0') }}</span>
                <span class="ball-sep">+</span>
                <span
                  v-for="n in win.winning_backs"
                  :key="'wb-' + n"
                  class="ball back-ball"
                >{{ String(n).padStart(2, '0') }}</span>
              </div>
            </div>
            <div class="win-row">
              <span class="win-label">对比期号</span>
              <span class="win-value">第 {{ win.issue }} 期（{{ win.date }}）</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 实时日志面板 -->
    <GlassCard v-if="isRunning || logLines.length > 0">
      <div class="page-content">
        <div class="log-header">
          <h3 class="section-title">执行日志</h3>
          <div class="log-header-actions">
            <label class="auto-scroll-toggle">
              <input type="checkbox" v-model="autoScroll" />
              自动滚动
            </label>
            <button class="btn-text" @click="clearLogs">清空</button>
          </div>
        </div>
        <div class="log-panel" ref="logPanel">
          <div
            v-for="(line, i) in logLines"
            :key="i"
            class="log-line"
            :class="'log-' + line.level"
          >
            <span class="log-ts">{{ line.time }}</span>
            <span class="log-msg">{{ line.message }}</span>
          </div>
          <div v-if="isRunning" class="log-line log-cursor">
            <span class="cursor-dot"></span>
          </div>
        </div>
      </div>
    </GlassCard>

    <!-- 完成后汇总统计 -->
    <template v-if="runResult">
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value">{{ runResult.stats.total_tests }}</div>
          <div class="stat-label">总测试次数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value win-count">{{ runResult.stats.winning_tests }}</div>
          <div class="stat-label">中奖次数</div>
        </div>
        <div class="stat-card">
          <div class="stat-value highlight-rate">{{ formatRate(runResult.stats.winning_rate) }}</div>
          <div class="stat-label">总中奖率</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">{{ runResult.successful_methods.length }}</div>
          <div class="stat-label">达标方法数</div>
        </div>
      </div>

      <GlassCard>
        <div class="page-content">
          <h3 class="section-title">测试摘要</h3>
          <div class="summary-lines">
            <div>会话ID：{{ runResult.session_id }}</div>
            <div>测试策略：{{ runResult.strategy }}</div>
            <div>目标奖级：{{ runResult.target_prize }}</div>
          </div>

          <div class="section-divider"></div>

          <h3 class="section-title">达标方法</h3>
          <div class="chip-list" v-if="runResult.successful_methods.length > 0">
            <span
              v-for="method in runResult.successful_methods"
              :key="method"
              class="chip chip-success"
            >{{ methodLabel(method) }}</span>
          </div>
          <div v-else class="empty-tip">当前无达到目标奖级的方法</div>

          <div class="section-divider"></div>

          <h3 class="section-title">奖级分布</h3>
          <div class="chip-list">
            <span v-for="prize in orderedPrizeLevels" :key="prize" class="chip">
              {{ prize }}: {{ runResult.stats.prize_stats[prize] || 0 }}
            </span>
          </div>
        </div>
      </GlassCard>

      <GlassCard>
        <div class="page-content">
          <h3 class="section-title">方法表现明细</h3>
          <div class="table-container">
            <table class="data-table">
              <thead>
                <tr>
                  <th>方法</th>
                  <th>测试次数</th>
                  <th>中奖次数</th>
                  <th>中奖率</th>
                  <th>最佳奖级</th>
                </tr>
              </thead>
              <tbody>
                <template v-for="item in methodStatRows" :key="item.method">
                  <tr
                    :class="{ 'row-expandable': methodWinnings(item.method).length > 0 }"
                    @click="toggleExpand(item.method)"
                  >
                    <td>
                      <span
                        v-if="methodWinnings(item.method).length > 0"
                        class="expand-icon"
                      >{{ expandedMethods.has(item.method) ? '▼' : '▶' }}</span>
                      {{ methodLabel(item.method) }}
                    </td>
                    <td>{{ item.total_tests }}</td>
                    <td>{{ item.winning_tests }}</td>
                    <td>{{ formatRate(item.winning_rate) }}</td>
                    <td :class="item.best_prize ? 'prize-cell' : ''">{{ item.best_prize || '未中奖' }}</td>
                  </tr>
                  <template v-if="expandedMethods.has(item.method)">
                  <tr
                    v-for="(win, wi) in methodWinnings(item.method)"
                    :key="item.method + '-win-' + wi"
                    class="winning-detail-row"
                  >
                    <td colspan="5">
                      <div class="winning-detail-cell">
                        <span class="detail-prize-tag" :class="prizeCls(win.prize_level)">{{ win.prize_level }}</span>
                        <span class="detail-label">预测:</span>
                        <span
                          v-for="n in win.predicted_fronts"
                          :key="'dp-' + n"
                          class="ball ball-sm front-ball"
                          :class="{ 'ball-hit': win.winning_fronts.includes(n) }"
                        >{{ String(n).padStart(2, '0') }}</span>
                        <span class="ball-sep-sm">+</span>
                        <span
                          v-for="n in win.predicted_backs"
                          :key="'dpb-' + n"
                          class="ball ball-sm back-ball"
                          :class="{ 'ball-hit': win.winning_backs.includes(n) }"
                        >{{ String(n).padStart(2, '0') }}</span>
                        <span class="detail-sep">|</span>
                        <span class="detail-label">开奖:</span>
                        <span
                          v-for="n in win.winning_fronts"
                          :key="'dw-' + n"
                          class="ball ball-sm front-ball"
                        >{{ String(n).padStart(2, '0') }}</span>
                        <span class="ball-sep-sm">+</span>
                        <span
                          v-for="n in win.winning_backs"
                          :key="'dwb-' + n"
                          class="ball ball-sm back-ball"
                        >{{ String(n).padStart(2, '0') }}</span>
                        <span class="detail-issue">第{{ win.issue }}期</span>
                      </div>
                    </td>
                  </tr>
                  </template>
              </tbody>
            </table>
          </div>
        </div>
      </GlassCard>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import GlassCard from '@/components/common/GlassCard.vue'
import { getTestingOptions, createTestingStream } from '@/api/testing'
import type { MethodInfo } from '@/types/testing'
import type {
  TestingRunParams,
  TestingRunResult,
  SseLogEvent,
  SseProgressEvent,
  SseWinningEvent
} from '@/types/testing'

interface MethodStatRow {
  method: string
  total_tests: number
  winning_tests: number
  winning_rate: number
  best_prize: string | null
}

interface LogLine {
  time: string
  message: string
  level: string
}

interface WinningRecord extends SseWinningEvent {
  timestamp: string
}

const loadingMethods = ref(false)
const isRunning = ref(false)
const runResult = ref<TestingRunResult | null>(null)
const methodGroups = ref<Record<string, MethodInfo[]>>({})
const selectedMethods = ref<string[]>([])
const currentProgress = ref<SseProgressEvent | null>(null)
const logLines = ref<LogLine[]>([])
const winningRecords = ref<WinningRecord[]>([])
const autoScroll = ref(true)
const logPanel = ref<HTMLElement | null>(null)
const liveStats = ref({ total: 0, winning: 0 })
const expandedMethods = ref<Set<string>>(new Set())
const prizeOptions = ref<string[]>([])

let eventSource: EventSource | null = null

const params = ref<TestingRunParams>({
  methods: [],
  strategy: 'progressive',
  target_prize: '六等奖',
  periods_start: 10,
  periods_end: 2000,
  count_start: 1,
  count_end: 5,
  max_tests: 50,
  parallel: false,
  workers: 4
})

const orderedPrizeLevels = ['一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖', '七等奖', '八等奖', '九等奖']

const allMethodIds = computed(() =>
  Object.values(methodGroups.value)
    .flat()
    .map((item) => item.id)
)

const isAllSelected = computed(() =>
  allMethodIds.value.length > 0 &&
  selectedMethods.value.length === allMethodIds.value.length
)

const methodNameMap = computed(() => {
  const mapping: Record<string, string> = {}
  Object.values(methodGroups.value)
    .flat()
    .forEach((method) => {
      mapping[method.id] = method.name
    })
  return mapping
})

const methodStatRows = computed<MethodStatRow[]>(() => {
  if (!runResult.value) return []
  return Object.entries(runResult.value.stats.method_stats || {})
    .map(([method, stat]) => ({
      method,
      total_tests: stat.tests || stat.total_tests || 0,
      winning_tests: stat.winning_tests || 0,
      winning_rate: stat.winning_rate || 0,
      best_prize: stat.best_prize_name || stat.best_prize || null
    }))
    .sort((a, b) => b.winning_rate - a.winning_rate)
})

const methodLabel = (id: string) => methodNameMap.value[id] || id
const formatRate = (value: number) => `${(value * 100).toFixed(2)}%`

const formatTime = (iso: string) => {
  try {
    const d = new Date(iso)
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
  } catch {
    return ''
  }
}

const prizeCls = (level: string) => {
  const map: Record<string, string> = {
    '一等奖': 'prize-1',
    '二等奖': 'prize-2',
    '三等奖': 'prize-3',
    '四等奖': 'prize-4',
    '五等奖': 'prize-5',
    '六等奖': 'prize-6',
    '七等奖': 'prize-7',
    '八等奖': 'prize-8',
    '九等奖': 'prize-9'
  }
  return map[level] || ''
}

const toggleExpand = (method: string) => {
  const s = expandedMethods.value
  if (s.has(method)) {
    s.delete(method)
  } else {
    s.add(method)
  }
  expandedMethods.value = new Set(s)
}

const methodWinnings = (method: string) => {
  return winningRecords.value.filter(w => w.method === method)
}

const scrollToBottom = () => {
  if (autoScroll.value && logPanel.value) {
    logPanel.value.scrollTop = logPanel.value.scrollHeight
  }
}

const pushLog = (message: string, level = 'info') => {
  const now = new Date()
  const time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`
  logLines.value.push({ time, message, level })
  if (logLines.value.length > 500) {
    logLines.value.splice(0, logLines.value.length - 500)
  }
  nextTick(scrollToBottom)
}

const clearLogs = () => {
  logLines.value = []
}

const loadMethods = async () => {
  loadingMethods.value = true
  try {
    const res = await getTestingOptions()
    if (!res.success || !res.data) return

    const methods = res.data.methods || []
    const groups: Record<string, MethodInfo[]> = {}
    methods.forEach((method: MethodInfo) => {
      const category = method.category || '其他'
      if (!groups[category]) groups[category] = []
      groups[category].push(method)
    })
    methodGroups.value = groups
    selectedMethods.value = allMethodIds.value.slice(0, Math.min(3, allMethodIds.value.length))

    prizeOptions.value = res.data.target_prizes || orderedPrizeLevels
  } catch (error) {
    console.error('加载方法失败:', error)
  } finally {
    loadingMethods.value = false
  }
}

const toggleSelectAll = () => {
  if (isAllSelected.value) {
    selectedMethods.value = []
  } else {
    selectedMethods.value = [...allMethodIds.value]
  }
}

const cleanupStream = () => {
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
  isRunning.value = false
  currentProgress.value = null
}

const stopTesting = () => {
  cleanupStream()
  pushLog('测试已手动停止', 'warning')
}

const startTesting = () => {
  if (selectedMethods.value.length === 0) return

  isRunning.value = true
  runResult.value = null
  currentProgress.value = null
  winningRecords.value = []
  logLines.value = []
  liveStats.value = { total: 0, winning: 0 }

  const req: TestingRunParams = {
    ...params.value,
    methods: [...selectedMethods.value]
  }

  pushLog(`开始测试，共 ${req.methods.length} 个方法 | 策略：${req.strategy} | 目标奖级：${req.target_prize}`)

  if (eventSource) {
    eventSource.close()
    eventSource = null
  }

  eventSource = createTestingStream(req)

  eventSource.addEventListener('log', (e: MessageEvent) => {
    try {
      const data: SseLogEvent = JSON.parse(e.data)
      pushLog(data.message, data.level)
    } catch {
      pushLog(e.data)
    }
  })

  eventSource.addEventListener('progress', (e: MessageEvent) => {
    try {
      const data: SseProgressEvent = JSON.parse(e.data)
      currentProgress.value = data
    } catch { /* 忽略 */ }
  })

  eventSource.addEventListener('result', (e: MessageEvent) => {
    try {
      const data = JSON.parse(e.data)
      liveStats.value.total++
      if (data.winning_count > 0) {
        liveStats.value.winning += data.winning_count
      }
    } catch { /* 忽略 */ }
  })

  eventSource.addEventListener('winning', (e: MessageEvent) => {
    try {
      const data: SseWinningEvent = JSON.parse(e.data)
      winningRecords.value.unshift({
        ...data,
        timestamp: new Date().toISOString()
      })
      pushLog(
        `中奖！方法：${methodLabel(data.method)} | 期数：${data.periods} 期 | 奖级：${data.prize_level}`,
        'warning'
      )
    } catch { /* 忽略 */ }
  })

  eventSource.addEventListener('complete', (e: MessageEvent) => {
    try {
      const data = JSON.parse(e.data)
      // 后端 summary 中字段在顶层，组装为前端需要的格式
      const stats = {
        total_tests: data.total_tests || 0,
        winning_tests: data.winning_tests || 0,
        winning_rate: data.winning_rate || 0,
        method_stats: data.method_stats || {},
        prize_stats: data.prize_stats || {}
      }
      if (stats.total_tests > 0) {
        runResult.value = {
          session_id: data.session_id || '',
          strategy: data.config?.strategy || req.strategy,
          target_prize: data.config?.target_prize || req.target_prize,
          tested_methods: data.tested_methods || req.methods,
          successful_methods: data.successful_methods || [],
          stats,
          report_files: data.report_files || { json: '', text: '' },
          time: data.test_time || new Date().toISOString()
        }
        pushLog(`测试完成！总测试 ${stats.total_tests} 次，中奖 ${stats.winning_tests} 次`)
      } else {
        pushLog(data.message || '测试结束', 'error')
      }
    } catch {
      pushLog('测试已结束', 'info')
    }
    cleanupStream()
  })

  eventSource.addEventListener('error_event', (e: MessageEvent) => {
    try {
      const data = JSON.parse(e.data)
      pushLog(data.message || '发生错误', 'error')
    } catch {
      pushLog('发生未知错误', 'error')
    }
    cleanupStream()
  })

  eventSource.onerror = () => {
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
    if (isRunning.value) {
      pushLog('SSE 连接断开', 'error')
      isRunning.value = false
      currentProgress.value = null
    }
  }
}

onMounted(() => {
  loadMethods()
})

onUnmounted(() => {
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
})
</script>

<style scoped>
.testing-page {
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.page-content {
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.page-title {
  margin: 0;
  font-size: 22px;
  color: var(--text-primary);
}

.header-actions {
  display: flex;
  gap: 12px;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 14px;
  margin-bottom: 20px;
}

.config-item label {
  display: block;
  margin-bottom: 6px;
  color: var(--text-secondary);
  font-size: 13px;
}

.config-item select,
.config-item input[type='number'] {
  width: 100%;
  padding: 10px;
  border: 1px solid var(--border-default);
  border-radius: 6px;
  background: var(--bg-primary);
  color: var(--text-primary);
}

.checkbox-item {
  display: flex;
  align-items: flex-end;
}

.checkbox-item label {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  cursor: pointer;
}

.range-inputs {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 8px;
}

.range-inputs span {
  color: var(--text-muted);
}

.method-section {
  border-top: 1px solid var(--border-default);
  padding-top: 16px;
}

.section-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.section-title {
  margin: 0;
  color: var(--text-primary);
  font-size: 16px;
}

.method-count {
  color: var(--text-muted);
  font-size: 13px;
}

.method-groups {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}

.method-group {
  padding: 12px;
  border: 1px solid var(--border-default);
  border-radius: 8px;
  background: var(--bg-secondary);
}

.group-title {
  color: var(--neon-blue);
  margin-bottom: 10px;
  font-weight: 600;
}

.method-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 180px;
  overflow: auto;
}

.method-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.method-item input {
  accent-color: var(--neon-blue);
}

/* 进度指示栏 */
.progress-banner {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  border-radius: 8px;
  background: rgba(0, 212, 255, 0.07);
  border: 1px solid rgba(0, 212, 255, 0.22);
}

.progress-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: var(--text-secondary);
}

.progress-stats {
  font-size: 13px;
  color: var(--text-muted);
  flex-shrink: 0;
}

.spinner {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid rgba(0, 212, 255, 0.25);
  border-top-color: var(--neon-blue);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  flex-shrink: 0;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 中奖区块 */
.winning-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.winning-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.winning-title {
  margin: 0;
  font-size: 18px;
  color: var(--text-primary);
}

.winning-badge {
  background: rgba(255, 200, 0, 0.12);
  color: #ffc800;
  border: 1px solid rgba(255, 200, 0, 0.35);
  border-radius: 999px;
  padding: 2px 10px;
  font-size: 12px;
  font-weight: 600;
}

.winning-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.winning-card {
  border-radius: 10px;
  border: 1px solid var(--border-default);
  background: var(--bg-secondary);
  overflow: hidden;
}

.win-card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-default);
}

.prize-tag {
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 700;
  background: rgba(255, 200, 0, 0.12);
  color: #ffc800;
  border: 1px solid rgba(255, 200, 0, 0.35);
}

.prize-1 .prize-tag { background: rgba(255, 60, 60, 0.12); color: #ff4444; border-color: rgba(255, 60, 60, 0.35); }
.prize-2 .prize-tag { background: rgba(255, 140, 0, 0.12); color: #ff8c00; border-color: rgba(255, 140, 0, 0.35); }
.prize-3 .prize-tag { background: rgba(255, 200, 0, 0.12); color: #ffc800; border-color: rgba(255, 200, 0, 0.35); }
.prize-4 .prize-tag { background: rgba(82, 196, 26, 0.12); color: #52c41a; border-color: rgba(82, 196, 26, 0.35); }
.prize-5 .prize-tag { background: rgba(0, 212, 255, 0.12); color: var(--neon-blue); border-color: rgba(0, 212, 255, 0.35); }
.prize-6 .prize-tag { background: rgba(160, 160, 160, 0.12); color: #aaa; border-color: rgba(160, 160, 160, 0.35); }
.prize-7 .prize-tag, .prize-8 .prize-tag, .prize-9 .prize-tag { background: rgba(100, 100, 120, 0.12); color: #888; border-color: rgba(100, 100, 120, 0.35); }

.win-method {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.win-time {
  margin-left: auto;
  font-size: 12px;
  color: var(--text-muted);
}

.win-card-body {
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.win-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.win-label {
  font-size: 12px;
  color: var(--text-muted);
  width: 60px;
  flex-shrink: 0;
}

.win-value {
  font-size: 13px;
  color: var(--text-secondary);
}

.balls-row {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}

.ball {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  font-size: 12px;
  font-weight: 700;
}

.front-ball {
  background: rgba(0, 212, 255, 0.12);
  color: var(--neon-blue);
  border: 1px solid rgba(0, 212, 255, 0.3);
}

.back-ball {
  background: rgba(255, 0, 255, 0.12);
  color: var(--neon-purple);
  border: 1px solid rgba(255, 0, 255, 0.3);
}

.ball-hit {
  box-shadow: 0 0 0 2px currentColor;
  font-weight: 900;
}

.ball-sep {
  color: var(--text-muted);
  font-size: 14px;
  font-weight: 600;
  margin: 0 2px;
}

.ball-sep-sm {
  color: var(--text-muted);
  font-size: 12px;
  margin: 0 1px;
}

/* 日志面板 */
.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.log-header-actions {
  display: flex;
  align-items: center;
  gap: 14px;
}

.auto-scroll-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-muted);
  cursor: pointer;
}

.log-panel {
  height: 320px;
  overflow-y: auto;
  background: #070b14;
  border-radius: 6px;
  padding: 10px 14px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.7;
  scroll-behavior: smooth;
}

.log-line {
  display: flex;
  gap: 10px;
}

.log-ts {
  color: #3d5060;
  flex-shrink: 0;
  user-select: none;
}

.log-info .log-msg { color: #7a8fa0; }
.log-warning .log-msg { color: #ffc800; }
.log-error .log-msg { color: #ff5555; }
.log-debug .log-msg { color: #4a7a99; }

.log-cursor {
  padding-top: 2px;
}

.cursor-dot {
  display: inline-block;
  width: 7px;
  height: 14px;
  background: var(--neon-blue);
  border-radius: 2px;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  50% { opacity: 0; }
}

/* 按钮 */
.btn-primary,
.btn-secondary,
.btn-danger,
.btn-text {
  border: none;
  border-radius: 6px;
  padding: 10px 16px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
}

.btn-primary {
  color: #fff;
  background: linear-gradient(135deg, var(--neon-blue), var(--neon-blue-dark));
}

.btn-secondary {
  color: var(--text-secondary);
  background: var(--bg-secondary);
  border: 1px solid var(--border-default);
}

.btn-danger {
  color: #fff;
  background: #c0392b;
}

.btn-text {
  color: var(--text-muted);
  background: transparent;
  padding: 4px 8px;
  font-size: 12px;
}

.btn-primary:disabled,
.btn-secondary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* 汇总统计 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 14px;
}

.stat-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-default);
  border-radius: 10px;
  padding: 16px;
  text-align: center;
}

.stat-value {
  font-size: 24px;
  color: var(--text-primary);
  font-weight: 700;
}

.highlight-rate {
  color: var(--neon-blue);
}

.win-count {
  color: #ffc800;
}

.stat-label {
  margin-top: 6px;
  color: var(--text-muted);
  font-size: 13px;
}

.summary-lines {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.section-divider {
  height: 1px;
  margin: 16px 0;
  background: var(--border-default);
}

.chip-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.chip {
  border: 1px solid var(--border-default);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  color: var(--text-secondary);
  background: var(--bg-secondary);
}

.chip-success {
  color: var(--neon-green);
  border-color: rgba(0, 255, 136, 0.4);
  background: rgba(0, 255, 136, 0.1);
}

.empty-tip {
  color: var(--text-muted);
  font-size: 13px;
}

.table-container {
  overflow: auto;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.data-table th,
.data-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid var(--border-default);
  color: var(--text-secondary);
}

.data-table th {
  color: var(--text-primary);
  font-weight: 600;
}

.prize-cell {
  color: #ffc800;
  font-weight: 600;
}

/* 方法明细表展开行 */
.row-expandable {
  cursor: pointer;
}
.row-expandable:hover {
  background: rgba(0, 212, 255, 0.05);
}
.expand-icon {
  font-size: 10px;
  margin-right: 4px;
  color: var(--neon-blue);
}
.winning-detail-row td {
  background: rgba(0, 40, 60, 0.5);
  border-top: none;
  padding: 6px 12px;
}
.winning-detail-cell {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  padding-left: 20px;
}
.detail-prize-tag {
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 4px;
  font-weight: 600;
  color: #fff;
  background: #555;
}
.detail-prize-tag.prize-1 { background: #e53935; }
.detail-prize-tag.prize-2 { background: #fb8c00; }
.detail-prize-tag.prize-3 { background: #fdd835; color: #333; }
.detail-prize-tag.prize-4 { background: #43a047; }
.detail-prize-tag.prize-5 { background: #1e88e5; }
.detail-prize-tag.prize-6 { background: #8e24aa; }
.detail-prize-tag.prize-7, .detail-prize-tag.prize-8, .detail-prize-tag.prize-9 { background: #616161; }
.detail-label {
  font-size: 12px;
  color: #8899aa;
  margin-left: 4px;
}
.detail-sep {
  color: #445566;
  margin: 0 4px;
}
.detail-issue {
  font-size: 11px;
  color: #667788;
  margin-left: 8px;
}
.ball-sm {
  width: 24px;
  height: 24px;
  font-size: 11px;
  line-height: 24px;
}

@media (max-width: 768px) {
  .testing-page {
    padding: 14px;
  }

  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .header-actions {
    width: 100%;
  }

  .btn-primary,
  .btn-secondary,
  .btn-danger {
    flex: 1;
  }

  .progress-banner {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
}
</style>
