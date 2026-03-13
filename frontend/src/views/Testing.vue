<template>
  <div class="testing-page">
    <GlassCard title="测试参数">
      <div class="form-grid">
        <div class="form-item methods-item">
          <label>预测方法</label>
          <n-select
            v-model:value="form.methods"
            :options="methodOptions"
            multiple
            filterable
            placeholder="选择一个或多个方法"
          />
        </div>
        <div class="form-item">
          <label>策略</label>
          <n-radio-group v-model:value="form.strategy" name="strategy">
            <n-space>
              <n-radio-button value="random">随机</n-radio-button>
              <n-radio-button value="progressive">渐进</n-radio-button>
            </n-space>
          </n-radio-group>
        </div>
        <div class="form-item">
          <label>目标奖级</label>
          <n-select v-model:value="form.target_prize" :options="prizeOptions" />
        </div>
        <div class="form-item">
          <label>分析期数范围</label>
          <div class="inline-range">
            <n-input-number v-model:value="form.periods_start" :min="10" :max="2748" />
            <span>到</span>
            <n-input-number v-model:value="form.periods_end" :min="10" :max="2748" />
          </div>
        </div>
        <div class="form-item">
          <label>注数范围</label>
          <div class="inline-range">
            <n-input-number v-model:value="form.count_start" :min="1" :max="100" />
            <span>到</span>
            <n-input-number v-model:value="form.count_end" :min="1" :max="100" />
          </div>
        </div>
        <div class="form-item">
          <label>每方法最大测试次数</label>
          <n-input-number v-model:value="form.max_tests" :min="1" :max="5000" />
        </div>
        <div class="form-item">
          <label>并行与线程</label>
          <div class="inline-range">
            <n-switch v-model:value="form.parallel" />
            <n-input-number v-model:value="form.workers" :min="1" :max="64" :disabled="!form.parallel" />
          </div>
        </div>
        <div class="form-item">
          <label>随机种子（可选）</label>
          <n-input-number v-model:value="seedValue" :min="0" :max="999999" />
        </div>
      </div>

      <div class="actions-row">
        <n-button type="primary" :loading="running" @click="startTesting">开始测试</n-button>
        <n-button type="warning" :disabled="!running" @click="stopTesting">停止</n-button>
        <n-button @click="resetForm">重置</n-button>
      </div>
    </GlassCard>

    <GlassCard v-if="running || progress.total > 0" title="实时进度" glow>
      <n-progress type="line" :percentage="progress.percent" :height="12" indicator-placement="inside" />
      <div class="progress-meta">
        <span>当前: {{ progress.current }} / {{ progress.total }}</span>
        <span>方法: {{ progress.method || '-' }}</span>
      </div>
    </GlassCard>

    <div class="info-grid">
      <GlassCard title="中奖事件">
        <div v-if="winningEvents.length === 0" class="empty-text">暂无中奖事件</div>
        <div v-else class="event-list">
          <div v-for="(item, idx) in winningEvents" :key="idx" class="event-item">
            <span class="event-title">{{ item.method }} - {{ item.prize_name }}</span>
            <span class="event-desc">{{ item.match_combination }} | p={{ item.periods }} c={{ item.count }}</span>
          </div>
        </div>
      </GlassCard>

      <GlassCard title="实时日志">
        <div v-if="logs.length === 0" class="empty-text">暂无日志</div>
        <div v-else class="log-list">
          <div v-for="(line, idx) in logs" :key="idx" class="log-line">{{ line }}</div>
        </div>
      </GlassCard>
    </div>

    <GlassCard v-if="summary" title="测试汇总">
      <div class="summary-grid">
        <div class="summary-item">
          <span class="label">Session</span>
          <span class="value">{{ summary.session_id }}</span>
        </div>
        <div class="summary-item">
          <span class="label">总测试数</span>
          <span class="value">{{ summary.total_tests }}</span>
        </div>
        <div class="summary-item">
          <span class="label">中奖测试</span>
          <span class="value">{{ summary.winning_tests }}</span>
        </div>
        <div class="summary-item">
          <span class="label">中奖率</span>
          <span class="value">{{ formatPercent(summary.winning_rate) }}</span>
        </div>
      </div>

      <n-data-table :columns="outcomeColumns" :data="summary.method_outcomes || []" :bordered="false" size="small" />

      <div class="report-files" v-if="summary.report_files">
        <span>报告文件：</span>
        <span>{{ summary.report_files.json }}</span>
      </div>
    </GlassCard>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useMessage } from 'naive-ui'
import type { DataTableColumns } from 'naive-ui'
import GlassCard from '@/components/common/GlassCard.vue'
import { createTestingStreamUrl, getTestingOptions } from '@/api/testing'
import type { TestingEvent, TestingRequest, TestingSummary } from '@/types/testing'

const message = useMessage()

const methodOptions = ref<Array<{ label: string; value: string }>>([])
const prizeOptions = ref<Array<{ label: string; value: string }>>([])

const form = reactive<TestingRequest>({
  methods: [],
  strategy: 'random',
  target_prize: '六等奖',
  periods_start: 50,
  periods_end: 500,
  count_start: 1,
  count_end: 1,
  max_tests: 20,
  parallel: false,
  workers: 4,
  seed: null,
  progressive_step: 50,
  timeout_seconds: 120,
  retries: 1
})

const seedValue = ref<number | null>(null)
const running = ref(false)
const logs = ref<string[]>([])
const winningEvents = ref<Array<Record<string, any>>>([])
const summary = ref<TestingSummary | null>(null)
const progress = reactive({ current: 0, total: 0, percent: 0, method: '' })

let stream: EventSource | null = null

const outcomeColumns: DataTableColumns = [
  { title: '方法', key: 'method' },
  { title: '测试次数', key: 'tests_run', width: 110 },
  { title: '是否达标', key: 'hit_target', width: 100, render: (row: any) => (row.hit_target ? '是' : '否') },
  { title: '最佳奖级', key: 'best_prize_name', width: 120 },
  { title: '错误', key: 'error' }
]

const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`

const resetForm = () => {
  form.methods = []
  form.strategy = 'random'
  form.target_prize = '六等奖'
  form.periods_start = 50
  form.periods_end = 500
  form.count_start = 1
  form.count_end = 1
  form.max_tests = 20
  form.parallel = false
  form.workers = 4
  form.progressive_step = 50
  form.timeout_seconds = 120
  form.retries = 1
  seedValue.value = null
  logs.value = []
  winningEvents.value = []
  summary.value = null
  progress.current = 0
  progress.total = 0
  progress.percent = 0
  progress.method = ''
}

const closeStream = () => {
  if (stream) {
    stream.close()
    stream = null
  }
}

const stopTesting = () => {
  closeStream()
  running.value = false
  message.info('已停止流式测试连接')
}

const handleEvent = (event: TestingEvent) => {
  const payload = event.data || {}
  if (event.type === 'log') {
    if (payload.message && payload.message !== 'heartbeat') {
      logs.value.unshift(String(payload.message))
      logs.value = logs.value.slice(0, 200)
    }
  } else if (event.type === 'progress') {
    progress.current = Number(payload.current || 0)
    progress.total = Number(payload.total || 0)
    progress.percent = Number(payload.percent || 0)
    progress.method = String(payload.method || '')
  } else if (event.type === 'winning') {
    winningEvents.value.unshift(payload)
    winningEvents.value = winningEvents.value.slice(0, 100)
  } else if (event.type === 'complete') {
    summary.value = payload as TestingSummary
    running.value = false
    closeStream()
    message.success('测试完成')
  } else if (event.type === 'error') {
    running.value = false
    closeStream()
    message.error(payload.message || '测试失败')
  }
}

const startTesting = () => {
  if (running.value) return
  if (!form.methods || form.methods.length === 0) {
    message.warning('请至少选择一个预测方法')
    return
  }
  if (form.periods_start > form.periods_end) {
    message.warning('分析期数范围不合法')
    return
  }
  if (form.count_start > form.count_end) {
    message.warning('注数范围不合法')
    return
  }

  logs.value = []
  winningEvents.value = []
  summary.value = null
  progress.current = 0
  progress.total = 0
  progress.percent = 0

  form.seed = typeof seedValue.value === 'number' ? seedValue.value : null

  const streamUrl = createTestingStreamUrl(form)
  stream = new EventSource(streamUrl)
  running.value = true

  stream.onmessage = (evt) => {
    try {
      const parsed = JSON.parse(evt.data) as TestingEvent
      handleEvent(parsed)
    } catch {
      // 忽略非 JSON 数据
    }
  }

  stream.onerror = () => {
    if (running.value) {
      message.error('测试连接中断')
      running.value = false
      closeStream()
    }
  }
}

onMounted(async () => {
  try {
    const resp = await getTestingOptions()
    if (resp.success && resp.data) {
      methodOptions.value = (resp.data.available_methods || []).map((m: string) => ({ label: m, value: m }))
      prizeOptions.value = (resp.data.target_prizes || []).map((p: string) => ({ label: p, value: p }))
      const firstOption = methodOptions.value[0]
      if (firstOption) {
        form.methods = [firstOption.value]
      }
    }
  } catch {
    message.error('加载测试选项失败')
  }
})

onBeforeUnmount(() => {
  closeStream()
})
</script>

<style scoped>
.testing-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px 16px;
}

.form-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.methods-item {
  grid-column: 1 / span 2;
}

.inline-range {
  display: flex;
  align-items: center;
  gap: 8px;
}

.actions-row {
  margin-top: 14px;
  display: flex;
  gap: 10px;
}

.progress-meta {
  margin-top: 8px;
  display: flex;
  justify-content: space-between;
  color: rgba(255, 255, 255, 0.7);
  font-size: 12px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.event-list,
.log-list {
  max-height: 220px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.event-item,
.log-line {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 8px;
  padding: 8px 10px;
}

.event-title {
  display: block;
  font-weight: 600;
}

.event-desc {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.65);
}

.empty-text {
  color: rgba(255, 255, 255, 0.5);
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}

.summary-item {
  padding: 10px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.summary-item .label {
  display: block;
  color: rgba(255, 255, 255, 0.6);
  font-size: 12px;
}

.summary-item .value {
  display: block;
  font-size: 14px;
  margin-top: 6px;
}

.report-files {
  margin-top: 10px;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.8);
}

@media (max-width: 960px) {
  .form-grid,
  .info-grid,
  .summary-grid {
    grid-template-columns: 1fr;
  }

  .methods-item {
    grid-column: auto;
  }
}
</style>
