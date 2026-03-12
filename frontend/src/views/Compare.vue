<template>
  <div class="compare-page">
    <!-- 对比配置 -->
    <GlassCard title="对比配置">
      <div class="compare-config">
        <div class="config-row">
          <div class="config-item">
            <label>目标期号</label>
            <n-input
              v-model:value="config.targetIssue"
              placeholder="输入要验证的期号"
            />
          </div>
          <div class="config-item">
            <label>预测算法</label>
            <n-select
              v-model:value="config.method"
              :options="algorithmOptions"
              placeholder="选择算法"
            />
          </div>
          <div class="config-item">
            <label>对比次数</label>
            <n-input-number
              v-model:value="config.times"
              :min="1"
              :max="1000"
              placeholder="对比次数"
            />
          </div>
        </div>
        <div class="config-row">
          <div class="config-item">
            <label>分析期数</label>
            <n-input-number
              v-model:value="config.periods"
              :min="50"
              :max="2000"
              placeholder="分析期数"
            />
          </div>
          <div class="config-item">
            <label>随机期数</label>
            <n-switch v-model:value="config.randomPeriods" />
          </div>
          <div class="config-item" v-if="config.randomPeriods">
            <label>期数范围</label>
            <div style="display: flex; gap: 8px;">
              <n-input-number
                v-model:value="config.minPeriods"
                :min="10"
                :max="config.maxPeriods"
                placeholder="最小"
                style="width: 100px"
              />
              <span style="line-height: 34px;">-</span>
              <n-input-number
                v-model:value="config.maxPeriods"
                :min="config.minPeriods"
                :max="2000"
                placeholder="最大"
                style="width: 100px"
              />
            </div>
          </div>
        </div>
        <div class="config-actions">
          <n-button type="primary" :loading="isComparing" @click="startCompare">
            <template #icon><GitCompareOutline /></template>
            开始对比
          </n-button>
          <n-button @click="resetConfig">
            <template #icon><RefreshOutline /></template>
            重置
          </n-button>
        </div>
      </div>
    </GlassCard>

    <!-- 对比进度 -->
    <GlassCard v-if="isComparing" title="对比进度" glow>
      <div class="compare-progress">
        <n-progress
          type="line"
          :percentage="compareProgress"
          :height="12"
          :border-radius="6"
          :fill-border-radius="6"
          indicator-placement="inside"
        />
        <span class="progress-text">
          正在对比第 {{ currentCompareIndex }} / {{ totalCompareCount }} 期...
        </span>
      </div>
    </GlassCard>

    <!-- 对比结果统计 -->
    <div v-if="compareResults.length > 0" class="results-stats">
      <GlassCard hoverable>
        <div class="stat-card">
          <div class="stat-icon primary">
            <LayersOutline />
          </div>
          <div class="stat-info">
            <span class="stat-value">{{ compareResults.length }}</span>
            <span class="stat-label">对比期数</span>
          </div>
        </div>
      </GlassCard>
      <GlassCard hoverable>
        <div class="stat-card">
          <div class="stat-icon success">
            <CheckmarkCircleOutline />
          </div>
          <div class="stat-info">
            <span class="stat-value">{{ hitStats.frontHit }}%</span>
            <span class="stat-label">前区命中率</span>
          </div>
        </div>
      </GlassCard>
      <GlassCard hoverable>
        <div class="stat-card">
          <div class="stat-icon warning">
            <CheckmarkCircleOutline />
          </div>
          <div class="stat-info">
            <span class="stat-value">{{ hitStats.backHit }}%</span>
            <span class="stat-label">后区命中率</span>
          </div>
        </div>
      </GlassCard>
      <GlassCard hoverable>
        <div class="stat-card">
          <div class="stat-icon purple">
            <TrophyOutline />
          </div>
          <div class="stat-info">
            <span class="stat-value">{{ hitStats.prizeCount }}</span>
            <span class="stat-label">中奖次数</span>
          </div>
        </div>
      </GlassCard>
    </div>

    <!-- 命中率趋势图 -->
    <GlassCard v-if="compareResults.length > 0" title="命中率趋势">
      <div class="hit-trend-chart" ref="hitTrendChartRef"></div>
    </GlassCard>

    <!-- 对比结果详情 -->
    <GlassCard v-if="compareResults.length > 0" title="对比详情" class="results-detail">
      <template #headerExtra>
        <n-button text type="primary" @click="exportResults">
          <template #icon><DownloadOutline /></template>
          导出
        </n-button>
      </template>

      <div class="results-table">
        <n-data-table
          :columns="resultColumns"
          :data="compareResults"
          :pagination="pagination"
          :bordered="false"
          :row-class-name="getRowClassName"
          size="small"
        />
      </div>
    </GlassCard>

    <!-- 算法排名 -->
    <GlassCard v-if="compareResults.length > 0" title="算法表现排名">
      <div class="algorithm-ranking">
        <div v-for="(algo, idx) in algorithmRanking" :key="algo.name" class="ranking-item">
          <div class="rank-badge" :class="getRankClass(idx)">{{ idx + 1 }}</div>
          <div class="algo-info">
            <span class="algo-name">{{ algo.name }}</span>
            <n-progress
              type="line"
              :percentage="algo.hitRate"
              :height="8"
              :show-indicator="false"
              :color="getProgressColor(algo.hitRate)"
            />
          </div>
          <div class="algo-stats">
            <span class="hit-rate">{{ algo.hitRate }}%</span>
            <span class="hit-count">命中 {{ algo.hitCount }} 次</span>
          </div>
        </div>
      </div>
    </GlassCard>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, h, onMounted, onUnmounted } from 'vue'
import { useMessage } from 'naive-ui'
import * as echarts from 'echarts'
import GlassCard from '@/components/common/GlassCard.vue'
import LotteryBall from '@/components/common/LotteryBall.vue'
import {
  GitCompareOutline,
  RefreshOutline,
  LayersOutline,
  CheckmarkCircleOutline,
  TrophyOutline,
  DownloadOutline
} from '@vicons/ionicons5'
import type { DataTableColumns } from 'naive-ui'
import { executeBatchComparison } from '@/api/compare'
import { getLatestData } from '@/api/data'

const message = useMessage()

// 对比配置
const config = ref({
  targetIssue: '',
  method: 'markov',
  periods: 100,
  times: 50,
  randomPeriods: false,
  minPeriods: 20,
  maxPeriods: 500
})

// 算法选项 - 全量算法支持
const algorithmOptions = [
  // 传统统计方法
  { label: '频率分析', value: 'frequency' },
  { label: '冷热分析', value: 'hot_cold' },
  { label: '遗漏分析', value: 'missing' },
  { label: '贝叶斯分析', value: 'bayesian' },
  // 马尔可夫链方法
  { label: '一阶马尔可夫', value: 'markov' },
  { label: '二阶马尔可夫', value: 'markov_2nd' },
  { label: '三阶马尔可夫', value: 'markov_3rd' },
  { label: '自适应马尔可夫', value: 'adaptive_markov' },
  { label: '自定义马尔可夫', value: 'markov_custom' },
  { label: '马尔可夫复式', value: 'markov_compound' },
  // 高级算法
  { label: '集成预测', value: 'ensemble' },
  { label: '聚类预测', value: 'clustering' },
  { label: '交集递减', value: 'consensus_halving' },
  { label: '九模型融合', value: 'nine_models' },
  { label: '九模型复式', value: 'nine_models_compound' },
  // 智能增强方法
  { label: '超级预测', value: 'super' },
  { label: '自适应预测', value: 'adaptive' },
  { label: '混合策略', value: 'mixed_strategy' },
  { label: '高度集成', value: 'highly_integrated' },
  { label: '高级集成', value: 'advanced_integration' },
  { label: '增强预测', value: 'enhanced' },
  // 集成学习方法
  { label: '堆叠集成', value: 'stacking' },
  { label: '自适应集成', value: 'adaptive_ensemble' },
  { label: '终极集成', value: 'ultimate_ensemble' },
  // 复式投注
  { label: '复式投注', value: 'compound' },
  { label: '胆拖投注', value: 'duplex' },
  // 深度学习方法
  { label: 'LSTM时序预测', value: 'lstm' },
  { label: 'Transformer', value: 'transformer' },
  { label: 'GAN生成对抗', value: 'gan' }
]

// 对比状态
const isComparing = ref(false)
const compareProgress = ref(0)
const currentCompareIndex = ref(0)
const totalCompareCount = computed(() => config.value.times)

// 对比结果
const compareResults = ref<Array<{
  period: string
  actual: { front: number[]; back: number[] }
  predicted: { front: number[]; back: number[] }
  frontHit: number
  backHit: number
  prizeLevel: string
  algorithm: string
}>>([])

// 命中统计
const hitStats = computed(() => {
  if (compareResults.value.length === 0) {
    return { frontHit: 0, backHit: 0, prizeCount: 0 }
  }
  const totalFrontHit = compareResults.value.reduce((acc, r) => acc + r.frontHit, 0)
  const totalBackHit = compareResults.value.reduce((acc, r) => acc + r.backHit, 0)
  const prizeCount = compareResults.value.filter(r => r.prizeLevel !== '-').length

  return {
    frontHit: Math.round((totalFrontHit / (compareResults.value.length * 5)) * 100),
    backHit: Math.round((totalBackHit / (compareResults.value.length * 2)) * 100),
    prizeCount
  }
})

// 算法排名
const algorithmRanking = computed(() => {
  const algoMap = new Map<string, { hitCount: number; total: number }>()

  compareResults.value.forEach(r => {
    const current = algoMap.get(r.algorithm) || { hitCount: 0, total: 0 }
    current.hitCount += r.frontHit + r.backHit
    current.total += 7
    algoMap.set(r.algorithm, current)
  })

  return Array.from(algoMap.entries())
    .map(([name, stats]) => ({
      name: algorithmOptions.find(a => a.value === name)?.label || name,
      hitRate: Math.round((stats.hitCount / stats.total) * 100),
      hitCount: stats.hitCount
    }))
    .sort((a, b) => b.hitRate - a.hitRate)
})

// 图表
const hitTrendChartRef = ref<HTMLElement | null>(null)
let hitTrendChart: echarts.ECharts | null = null

// 分页
const pagination = ref({
  page: 1,
  pageSize: 10,
  showSizePicker: true,
  pageSizes: [10, 20, 50, 100]
})

// 表格列定义
const resultColumns: DataTableColumns = [
  { title: '期号', key: 'period', width: 100, align: 'center' },
  {
    title: '实际号码',
    key: 'actual',
    width: 200,
    render(row: any) {
      return h('div', { class: 'ball-cell' }, [
        ...row.actual.front.map((n: number) =>
          h('span', { class: 'mini-ball front' }, String(n).padStart(2, '0'))
        ),
        h('span', { class: 'ball-sep' }, '+'),
        ...row.actual.back.map((n: number) =>
          h('span', { class: 'mini-ball back' }, String(n).padStart(2, '0'))
        )
      ])
    }
  },
  {
    title: '预测号码',
    key: 'predicted',
    width: 200,
    render(row: any) {
      return h('div', { class: 'ball-cell' }, [
        ...row.predicted.front.map((n: number, idx: number) =>
          h(
            'span',
            {
              class: [
                'mini-ball',
                'front',
                row.actual.front.includes(n) ? 'hit' : ''
              ]
            },
            String(n).padStart(2, '0')
          )
        ),
        h('span', { class: 'ball-sep' }, '+'),
        ...row.predicted.back.map((n: number) =>
          h(
            'span',
            {
              class: [
                'mini-ball',
                'back',
                row.actual.back.includes(n) ? 'hit' : ''
              ]
            },
            String(n).padStart(2, '0')
          )
        )
      ])
    }
  },
  {
    title: '前区命中',
    key: 'frontHit',
    width: 80,
    align: 'center',
    render(row: any) {
      const color = row.frontHit >= 3 ? '#00ff88' : row.frontHit >= 1 ? '#00d4ff' : '#ff4757'
      return h('span', { style: { color, fontWeight: 'bold' } }, row.frontHit)
    }
  },
  {
    title: '后区命中',
    key: 'backHit',
    width: 80,
    align: 'center',
    render(row: any) {
      const color = row.backHit >= 2 ? '#00ff88' : row.backHit >= 1 ? '#00d4ff' : '#ff4757'
      return h('span', { style: { color, fontWeight: 'bold' } }, row.backHit)
    }
  },
  {
    title: '中奖等级',
    key: 'prizeLevel',
    width: 100,
    align: 'center',
    render(row: any) {
      if (row.prizeLevel === '-') {
        return h('span', { style: { color: 'rgba(255,255,255,0.4)' } }, '-')
      }
      return h('span', { style: { color: '#00ff88', fontWeight: 'bold' } }, row.prizeLevel)
    }
  },
  { title: '算法', key: 'algorithm', width: 120, align: 'center' }
]

// 获取行样式
const getRowClassName = (row: any) => {
  if (row.prizeLevel !== '-') return 'prize-row'
  return ''
}

// 开始对比（SSE 流式进度）
const startCompare = async () => {
  if (!config.value.targetIssue) {
    message.warning('请输入目标期号')
    return
  }

  isComparing.value = true
  compareProgress.value = 0
  currentCompareIndex.value = 0
  compareResults.value = []

  const payload: any = {
    target_issue: config.value.targetIssue,
    method: config.value.method,
    periods: config.value.periods,
    times: config.value.times,
    random_periods: config.value.randomPeriods,
    export_excel: false,
    show_progress: false
  }
  if (config.value.randomPeriods) {
    payload.min_periods = config.value.minPeriods
    payload.max_periods = config.value.maxPeriods
  }

  try {
    const resp = await fetch('/api/compare/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })

    if (!resp.ok) {
      message.error(`请求失败: ${resp.status}`)
      isComparing.value = false
      return
    }

    const reader = resp.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const jsonStr = line.slice(6).trim()
        if (!jsonStr) continue

        try {
          const event = JSON.parse(jsonStr)

          if (event.type === 'progress') {
            currentCompareIndex.value = event.current
            compareProgress.value = Math.round((event.current / event.total) * 100)
          } else if (event.type === 'complete') {
            const { records, summary } = event.data
            compareResults.value = (records || []).map((record: any) => ({
              period: config.value.targetIssue,
              actual: {
                front: record.actual_front || summary?.actual_numbers?.front_balls || [],
                back: record.actual_back || summary?.actual_numbers?.back_balls || []
              },
              predicted: {
                front: record.predicted_front || [],
                back: record.predicted_back || []
              },
              frontHit: record.front_hits || 0,
              backHit: record.back_hits || 0,
              prizeLevel: getPrizeLevelFromNumber(record.prize_level),
              algorithm: algorithmOptions.find(a => a.value === config.value.method)?.label || config.value.method
            }))
            compareProgress.value = 100
            currentCompareIndex.value = records?.length || 0
            message.success(`对比完成，共 ${records?.length || 0} 轮`)
            setTimeout(() => initHitTrendChart(), 100)
          } else if (event.type === 'error') {
            message.error(`对比失败: ${event.message}`)
          }
        } catch {
          // 忽略解析错误
        }
      }
    }
  } catch (error: any) {
    message.error(`对比出错: ${error.message || '未知错误'}`)
  } finally {
    isComparing.value = false
  }
}

// 根据中奖等级数字返回中文名称
const getPrizeLevelFromNumber = (level: number): string => {
  const levelMap: Record<number, string> = {
    1: '一等奖',
    2: '二等奖',
    3: '三等奖',
    4: '四等奖',
    5: '五等奖',
    6: '六等奖',
    0: '-'
  }
  return levelMap[level] || '-'
}

// 生成随机号码
const generateRandomNumbers = (count: number, min: number, max: number): number[] => {
  const numbers: number[] = []
  while (numbers.length < count) {
    const num = Math.floor(Math.random() * (max - min + 1)) + min
    if (!numbers.includes(num)) numbers.push(num)
  }
  return numbers.sort((a, b) => a - b)
}

// 获取中奖等级
const getPrizeLevel = (frontHit: number, backHit: number): string => {
  if (frontHit === 5 && backHit === 2) return '一等奖'
  if (frontHit === 5 && backHit === 1) return '二等奖'
  if (frontHit === 5 && backHit === 0) return '三等奖'
  if (frontHit === 4 && backHit === 2) return '三等奖'
  if (frontHit === 4 && backHit === 1) return '四等奖'
  if (frontHit === 3 && backHit === 2) return '四等奖'
  if (frontHit === 4 && backHit === 0) return '五等奖'
  if (frontHit === 3 && backHit === 1) return '五等奖'
  if (frontHit === 2 && backHit === 2) return '五等奖'
  if (frontHit === 3 && backHit === 0) return '六等奖'
  if (frontHit === 2 && backHit === 1) return '六等奖'
  if (frontHit === 1 && backHit === 2) return '六等奖'
  if (frontHit === 0 && backHit === 2) return '六等奖'
  return '-'
}

// 重置配置
const resetConfig = async () => {
  // 获取最新期号作为默认目标期号
  try {
    const response = await getLatestData()
    if (response.success && response.data) {
      config.value.targetIssue = response.data.issue || ''
    }
  } catch (e) {
    // 忽略错误
  }
  config.value.method = 'markov'
  config.value.periods = 100
  config.value.times = 50
  config.value.randomPeriods = false
  config.value.minPeriods = 20
  config.value.maxPeriods = 500
  compareResults.value = []
}

// 获取排名样式
const getRankClass = (index: number) => {
  if (index === 0) return 'gold'
  if (index === 1) return 'silver'
  if (index === 2) return 'bronze'
  return ''
}

// 获取进度颜色
const getProgressColor = (value: number) => {
  if (value >= 50) return '#00ff88'
  if (value >= 30) return '#00d4ff'
  if (value >= 20) return '#ffaa00'
  return '#ff4757'
}

// 初始化趋势图
const initHitTrendChart = () => {
  if (!hitTrendChartRef.value || compareResults.value.length === 0) return

  hitTrendChart = echarts.init(hitTrendChartRef.value)

  const periods = compareResults.value.map(r => r.period)
  const frontHits = compareResults.value.map(r => r.frontHit)
  const backHits = compareResults.value.map(r => r.backHit)

  const option: echarts.EChartsOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 15, 35, 0.95)',
      borderColor: 'rgba(0, 212, 255, 0.3)',
      textStyle: { color: '#fff' }
    },
    legend: {
      data: ['前区命中', '后区命中'],
      textStyle: { color: 'rgba(255,255,255,0.6)' },
      top: 0
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '40px',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: periods,
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: {
        color: 'rgba(255,255,255,0.6)',
        fontSize: 10,
        interval: Math.floor(periods.length / 10)
      }
    },
    yAxis: {
      type: 'value',
      max: 5,
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: { color: 'rgba(255,255,255,0.6)' },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } }
    },
    series: [
      {
        name: '前区命中',
        type: 'line',
        smooth: true,
        data: frontHits,
        lineStyle: { color: '#00d4ff', width: 2 },
        itemStyle: { color: '#00d4ff' },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(0, 212, 255, 0.2)' },
            { offset: 1, color: 'rgba(0, 212, 255, 0)' }
          ])
        }
      },
      {
        name: '后区命中',
        type: 'line',
        smooth: true,
        data: backHits,
        lineStyle: { color: '#ff00ff', width: 2 },
        itemStyle: { color: '#ff00ff' }
      }
    ]
  }

  hitTrendChart.setOption(option)
}

// 导出结果
const exportResults = () => {
  const data = compareResults.value.map(r => ({
    期号: r.period,
    实际前区: r.actual.front.join(','),
    实际后区: r.actual.back.join(','),
    预测前区: r.predicted.front.join(','),
    预测后区: r.predicted.back.join(','),
    前区命中: r.frontHit,
    后区命中: r.backHit,
    中奖等级: r.prizeLevel,
    算法: r.algorithm
  }))

  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `compare_results_${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)
  message.success('导出成功')
}

// 响应式处理
const handleResize = () => {
  hitTrendChart?.resize()
}

onMounted(async () => {
  window.addEventListener('resize', handleResize)
  // 初始化时获取最新期号
  await resetConfig()
})

onUnmounted(() => {
  hitTrendChart?.dispose()
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.compare-page {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* 对比配置 */
.compare-config {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.config-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.config-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.config-item label {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
}

.config-actions {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
}

/* 对比进度 */
.compare-progress {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.progress-text {
  font-size: 14px;
  color: var(--text-secondary);
  text-align: center;
}

/* 结果统计 */
.results-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px;
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

.stat-icon.primary {
  background: rgba(0, 212, 255, 0.1);
  color: var(--neon-blue);
}

.stat-icon.success {
  background: rgba(0, 255, 136, 0.1);
  color: var(--neon-green);
}

.stat-icon.warning {
  background: rgba(255, 170, 0, 0.1);
  color: var(--neon-orange);
}

.stat-icon.purple {
  background: rgba(255, 0, 255, 0.1);
  color: var(--neon-purple);
}

.stat-info {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: var(--text-primary);
}

.stat-label {
  font-size: 13px;
  color: var(--text-secondary);
}

/* 趋势图 */
.hit-trend-chart {
  height: 300px;
}

/* 结果表格 */
.results-table {
  overflow-x: auto;
}

:deep(.ball-cell) {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-wrap: wrap;
}

:deep(.mini-ball) {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  font-size: 11px;
  font-weight: bold;
}

:deep(.mini-ball.front) {
  background: linear-gradient(135deg, #00d4ff, #0099cc);
  color: #0a0a1a;
}

:deep(.mini-ball.back) {
  background: linear-gradient(135deg, #ff00ff, #cc00cc);
  color: #0a0a1a;
}

:deep(.mini-ball.hit) {
  box-shadow: 0 0 8px rgba(0, 255, 136, 0.8);
  border: 2px solid #00ff88;
}

:deep(.ball-sep) {
  color: var(--text-tertiary);
  margin: 0 4px;
}

:deep(.prize-row) {
  background: rgba(0, 255, 136, 0.05) !important;
}

/* 算法排名 */
.algorithm-ranking {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.ranking-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: var(--radius-md);
}

.rank-badge {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 14px;
  background: rgba(255, 255, 255, 0.1);
  color: var(--text-secondary);
}

.rank-badge.gold {
  background: linear-gradient(135deg, #ffd700, #ffaa00);
  color: #0a0a1a;
}

.rank-badge.silver {
  background: linear-gradient(135deg, #c0c0c0, #a0a0a0);
  color: #0a0a1a;
}

.rank-badge.bronze {
  background: linear-gradient(135deg, #cd7f32, #8b4513);
  color: #0a0a1a;
}

.algo-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.algo-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.algo-stats {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}

.hit-rate {
  font-size: 18px;
  font-weight: bold;
  color: var(--neon-blue);
}

.hit-count {
  font-size: 12px;
  color: var(--text-tertiary);
}

/* 响应式 */
@media (max-width: 1200px) {
  .config-row {
    grid-template-columns: 1fr;
  }

  .results-stats {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
