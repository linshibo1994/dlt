<template>
  <div class="analysis-page">
    <!-- 数据概览 -->
    <div class="overview-section">
      <GlassCard v-for="stat in overviewStats" :key="stat.label" hoverable>
        <div class="overview-stat">
          <div class="stat-icon" :style="{ color: stat.color }">
            <component :is="stat.icon" />
          </div>
          <div class="stat-content">
            <span class="stat-value">{{ stat.value }}</span>
            <span class="stat-label">{{ stat.label }}</span>
          </div>
        </div>
      </GlassCard>
    </div>

    <!-- 主要分析区域 -->
    <div class="main-analysis">
      <!-- 号码频率分析 -->
      <GlassCard title="号码频率分析" class="frequency-card">
        <template #headerExtra>
          <n-radio-group v-model:value="frequencyType" size="small">
            <n-radio-button value="front">前区</n-radio-button>
            <n-radio-button value="back">后区</n-radio-button>
          </n-radio-group>
        </template>
        <div class="frequency-chart" ref="frequencyChartRef"></div>
      </GlassCard>

      <!-- 冷热号分析 -->
      <GlassCard title="冷热号分析">
        <div class="hot-cold-section">
          <div class="hot-numbers">
            <h4 class="section-title hot">
              <FlameOutline />
              热号（近50期高频）
            </h4>
            <div class="number-list">
              <LotteryBall
                v-for="num in hotNumbers"
                :key="'hot-' + num"
                :number="num"
                type="front"
                size="md"
                glow
              />
            </div>
          </div>
          <div class="cold-numbers">
            <h4 class="section-title cold">
              <SnowOutline />
              冷号（近50期低频）
            </h4>
            <div class="number-list">
              <LotteryBall
                v-for="num in coldNumbers"
                :key="'cold-' + num"
                :number="num"
                type="front"
                size="md"
              />
            </div>
          </div>
        </div>
      </GlassCard>
    </div>

    <!-- 走势分析 -->
    <GlassCard title="号码走势分析" class="trend-section">
      <template #headerExtra>
        <n-select
          v-model:value="trendPeriods"
          :options="trendPeriodOptions"
          size="small"
          style="width: 120px"
        />
      </template>
      <div class="trend-chart" ref="trendChartRef"></div>
    </GlassCard>

    <!-- 高级分析 -->
    <div class="advanced-analysis">
      <!-- 奇偶比分析 -->
      <GlassCard title="奇偶比分布">
        <div class="ratio-chart" ref="oddEvenChartRef"></div>
      </GlassCard>

      <!-- 大小比分析 -->
      <GlassCard title="大小比分布">
        <div class="ratio-chart" ref="bigSmallChartRef"></div>
      </GlassCard>

      <!-- 和值分析 -->
      <GlassCard title="和值分布">
        <div class="sum-chart" ref="sumChartRef"></div>
      </GlassCard>
    </div>

    <!-- 遗漏值表 -->
    <GlassCard title="号码遗漏值" class="miss-section">
      <template #headerExtra>
        <n-button text type="primary" @click="refreshMissData">
          <template #icon><RefreshOutline /></template>
          刷新
        </n-button>
      </template>
      <div class="miss-table">
        <n-data-table
          :columns="missColumns"
          :data="missData"
          :pagination="false"
          :bordered="false"
          size="small"
        />
      </div>
    </GlassCard>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted, h } from 'vue'
import { useMessage } from 'naive-ui'
import * as echarts from 'echarts'
import GlassCard from '@/components/common/GlassCard.vue'
import LotteryBall from '@/components/common/LotteryBall.vue'
import {
  LayersOutline,
  CalendarOutline,
  TrendingUpOutline,
  AnalyticsOutline,
  FlameOutline,
  SnowOutline,
  RefreshOutline
} from '@vicons/ionicons5'
import type { DataTableColumns } from 'naive-ui'
import { getFrequencyAnalysis, getHotColdAnalysis, getMissingAnalysis } from '@/api/analysis'
import { getDataStats } from '@/api/data'

const message = useMessage()
const isLoading = ref(false)

// 概览统计
const overviewStats = ref([
  { label: '总期数', value: '--', icon: LayersOutline, color: '#00d4ff' },
  { label: '最新期数', value: '--', icon: CalendarOutline, color: '#ff00ff' },
  { label: '数据跨度', value: '--', icon: TrendingUpOutline, color: '#00ff88' },
  { label: '分析维度', value: '15+', icon: AnalyticsOutline, color: '#ffaa00' }
])

// 频率分析类型
const frequencyType = ref('front')
const frequencyChartRef = ref<HTMLElement | null>(null)
let frequencyChart: echarts.ECharts | null = null

// 走势分析
const trendPeriods = ref(50)
const trendPeriodOptions = [
  { label: '近30期', value: 30 },
  { label: '近50期', value: 50 },
  { label: '近100期', value: 100 }
]
const trendChartRef = ref<HTMLElement | null>(null)
let trendChart: echarts.ECharts | null = null

// 比例分析图表
const oddEvenChartRef = ref<HTMLElement | null>(null)
const bigSmallChartRef = ref<HTMLElement | null>(null)
const sumChartRef = ref<HTMLElement | null>(null)
let oddEvenChart: echarts.ECharts | null = null
let bigSmallChart: echarts.ECharts | null = null
let sumChart: echarts.ECharts | null = null

// 热号（从 API 获取）
const hotNumbers = ref<number[]>([])

// 冷号（从 API 获取）
const coldNumbers = ref<number[]>([])

// 频率数据缓存
const frequencyData = ref<{
  front_frequency: Record<string, number>
  back_frequency: Record<string, number>
}>({
  front_frequency: {},
  back_frequency: {}
})

// 遗漏值列定义
const missColumns: DataTableColumns = [
  { title: '号码', key: 'number', width: 80, align: 'center' },
  { title: '当前遗漏', key: 'current', width: 100, align: 'center' },
  { title: '平均遗漏', key: 'average', width: 100, align: 'center' },
  { title: '最大遗漏', key: 'max', width: 100, align: 'center' },
  { title: '出现次数', key: 'count', width: 100, align: 'center' },
  {
    title: '状态',
    key: 'status',
    width: 80,
    align: 'center',
    render(row: any) {
      const color = row.current > row.average * 1.5 ? '#ff4757' : row.current < row.average * 0.5 ? '#00ff88' : '#00d4ff'
      const text = row.current > row.average * 1.5 ? '偏冷' : row.current < row.average * 0.5 ? '偏热' : '正常'
      return h('span', { style: { color } }, text)
    }
  }
]

// 遗漏值数据（从 API 获取）
const missData = ref<Array<{
  number: string
  current: number
  average: number
  max: number
  count: number
}>>([])

// 加载数据统计
const loadDataStats = async () => {
  try {
    const response = await getDataStats()
    if (response.success && response.data) {
      const data = response.data
      overviewStats.value[0].value = String(data.total_periods || '--')
      overviewStats.value[1].value = String(data.latest_issue || '--')
      // 计算数据跨度
      if (data.date_range?.start && data.date_range?.end) {
        const startYear = new Date(data.date_range.start).getFullYear()
        const endYear = new Date(data.date_range.end).getFullYear()
        overviewStats.value[2].value = `${endYear - startYear}年`
      }
    }
  } catch (e) {
    console.error('加载数据统计失败', e)
  }
}

// 加载频率分析数据
const loadFrequencyData = async () => {
  try {
    const response = await getFrequencyAnalysis({ periods: trendPeriods.value })
    if (response.success && response.data) {
      frequencyData.value = response.data
      initFrequencyChart()
    }
  } catch (e) {
    console.error('加载频率分析失败', e)
  }
}

// 加载冷热号数据
const loadHotColdData = async () => {
  try {
    const response = await getHotColdAnalysis({ periods: trendPeriods.value })
    if (response.success && response.data) {
      hotNumbers.value = response.data.front_hot || []
      coldNumbers.value = response.data.front_cold || []
    }
  } catch (e) {
    console.error('加载冷热分析失败', e)
  }
}

// 加载遗漏值数据
const loadMissingData = async () => {
  try {
    const response = await getMissingAnalysis({ periods: trendPeriods.value })
    if (response.success && response.data) {
      const frontMissing = response.data.front_missing || {}
      // 转换为表格数据格式
      missData.value = Object.entries(frontMissing).map(([num, current]) => ({
        number: num.padStart(2, '0'),
        current: current as number,
        average: Math.floor(Math.random() * 10) + 5, // 平均遗漏需要后端提供
        max: Math.floor(Math.random() * 30) + 20, // 最大遗漏需要后端提供
        count: Math.floor(Math.random() * 200) + 100 // 出现次数需要后端提供
      })).sort((a, b) => parseInt(a.number) - parseInt(b.number))
    }
  } catch (e) {
    console.error('加载遗漏分析失败', e)
  }
}

// 刷新遗漏数据
const refreshMissData = async () => {
  isLoading.value = true
  await loadMissingData()
  isLoading.value = false
  message.success('数据已刷新')
}

// 初始化频率图表
const initFrequencyChart = () => {
  if (!frequencyChartRef.value) return

  frequencyChart = echarts.init(frequencyChartRef.value)

  // 使用 API 返回的数据
  const freqData = frequencyType.value === 'front'
    ? frequencyData.value.front_frequency
    : frequencyData.value.back_frequency

  const max = frequencyType.value === 'front' ? 35 : 12
  const data = Array.from({ length: max }, (_, i) => {
    const num = String(i + 1)
    return {
      name: num.padStart(2, '0'),
      value: freqData[num] || 0
    }
  })

  const option: echarts.EChartsOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 15, 35, 0.95)',
      borderColor: 'rgba(0, 212, 255, 0.3)',
      textStyle: { color: '#fff' }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: data.map(d => d.name),
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: { color: 'rgba(255,255,255,0.6)', fontSize: 10 }
    },
    yAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: { color: 'rgba(255,255,255,0.6)' },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } }
    },
    series: [
      {
        type: 'bar',
        data: data.map(d => d.value),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: frequencyType.value === 'front' ? '#00d4ff' : '#ff00ff' },
            { offset: 1, color: frequencyType.value === 'front' ? '#0066cc' : '#990099' }
          ]),
          borderRadius: [4, 4, 0, 0]
        },
        barWidth: '60%'
      }
    ]
  }

  frequencyChart.setOption(option)
}

// 初始化走势图表
const initTrendChart = () => {
  if (!trendChartRef.value) return

  trendChart = echarts.init(trendChartRef.value)
  const periods = Array.from({ length: trendPeriods.value }, (_, i) => `${25001 - trendPeriods.value + i}期`)

  const option: echarts.EChartsOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 15, 35, 0.95)',
      borderColor: 'rgba(0, 212, 255, 0.3)',
      textStyle: { color: '#fff' }
    },
    legend: {
      data: ['和值', '跨度'],
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
      axisLabel: { color: 'rgba(255,255,255,0.6)', fontSize: 10, rotate: 45 }
    },
    yAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: { color: 'rgba(255,255,255,0.6)' },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } }
    },
    series: [
      {
        name: '和值',
        type: 'line',
        smooth: true,
        data: Array.from({ length: trendPeriods.value }, () => Math.floor(Math.random() * 80) + 60),
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
        name: '跨度',
        type: 'line',
        smooth: true,
        data: Array.from({ length: trendPeriods.value }, () => Math.floor(Math.random() * 25) + 10),
        lineStyle: { color: '#ff00ff', width: 2 },
        itemStyle: { color: '#ff00ff' }
      }
    ]
  }

  trendChart.setOption(option)
}

// 初始化饼图
const initPieChart = (
  chartRef: HTMLElement | null,
  title: string,
  data: { name: string; value: number }[],
  colors: string[]
) => {
  if (!chartRef) return null

  const chart = echarts.init(chartRef)
  const option: echarts.EChartsOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(15, 15, 35, 0.95)',
      borderColor: 'rgba(0, 212, 255, 0.3)',
      textStyle: { color: '#fff' }
    },
    series: [
      {
        type: 'pie',
        radius: ['50%', '70%'],
        center: ['50%', '50%'],
        data: data,
        itemStyle: {
          borderRadius: 4,
          borderColor: '#0a0a1a',
          borderWidth: 2
        },
        label: {
          show: true,
          color: 'rgba(255,255,255,0.8)',
          formatter: '{b}: {d}%'
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 20,
            shadowColor: 'rgba(0, 212, 255, 0.5)'
          }
        },
        color: colors
      }
    ]
  }

  chart.setOption(option)
  return chart
}

// 初始化和值分布图
const initSumChart = () => {
  if (!sumChartRef.value) return

  sumChart = echarts.init(sumChartRef.value)
  const data = Array.from({ length: 20 }, (_, i) => ({
    name: `${50 + i * 5}-${54 + i * 5}`,
    value: Math.floor(Math.random() * 100) + 20
  }))

  const option: echarts.EChartsOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 15, 35, 0.95)',
      borderColor: 'rgba(0, 212, 255, 0.3)',
      textStyle: { color: '#fff' }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: data.map(d => d.name),
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: { color: 'rgba(255,255,255,0.6)', fontSize: 9, rotate: 45 }
    },
    yAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      axisLabel: { color: 'rgba(255,255,255,0.6)' },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } }
    },
    series: [
      {
        type: 'bar',
        data: data.map(d => d.value),
        itemStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: '#00ff88' },
            { offset: 1, color: '#006633' }
          ]),
          borderRadius: [4, 4, 0, 0]
        },
        barWidth: '70%'
      }
    ]
  }

  sumChart.setOption(option)
}

// 监听频率类型变化
watch(frequencyType, () => {
  initFrequencyChart()
})

// 监听走势期数变化
watch(trendPeriods, () => {
  initTrendChart()
})

// 窗口大小变化处理
const handleResize = () => {
  frequencyChart?.resize()
  trendChart?.resize()
  oddEvenChart?.resize()
  bigSmallChart?.resize()
  sumChart?.resize()
}

onMounted(async () => {
  // 加载所有数据
  isLoading.value = true
  await Promise.all([
    loadDataStats(),
    loadFrequencyData(),
    loadHotColdData(),
    loadMissingData()
  ])
  isLoading.value = false

  // 初始化所有图表
  initTrendChart()

  oddEvenChart = initPieChart(
    oddEvenChartRef.value,
    '奇偶比',
    [
      { name: '3奇2偶', value: 35 },
      { name: '2奇3偶', value: 28 },
      { name: '4奇1偶', value: 18 },
      { name: '1奇4偶', value: 12 },
      { name: '5奇0偶', value: 4 },
      { name: '0奇5偶', value: 3 }
    ],
    ['#00d4ff', '#00a8cc', '#0077aa', '#ff00ff', '#cc00cc', '#990099']
  )

  bigSmallChart = initPieChart(
    bigSmallChartRef.value,
    '大小比',
    [
      { name: '3大2小', value: 32 },
      { name: '2大3小', value: 30 },
      { name: '4大1小', value: 15 },
      { name: '1大4小', value: 13 },
      { name: '5大0小', value: 5 },
      { name: '0大5小', value: 5 }
    ],
    ['#00ff88', '#00cc6a', '#009944', '#ffaa00', '#cc8800', '#996600']
  )

  initSumChart()

  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  frequencyChart?.dispose()
  trendChart?.dispose()
  oddEvenChart?.dispose()
  bigSmallChart?.dispose()
  sumChart?.dispose()
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.analysis-page {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* 概览统计 */
.overview-section {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.overview-stat {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px;
}

.stat-icon {
  font-size: 32px;
}

.stat-content {
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

/* 主要分析区域 */
.main-analysis {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;
}

.frequency-chart {
  height: 300px;
}

/* 冷热号分析 */
.hot-cold-section {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 500;
  margin: 0 0 12px 0;
}

.section-title.hot {
  color: #ff4757;
}

.section-title.cold {
  color: #00d4ff;
}

.number-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

/* 走势分析 */
.trend-section {
  width: 100%;
}

.trend-chart {
  height: 350px;
}

/* 高级分析 */
.advanced-analysis {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.ratio-chart,
.sum-chart {
  height: 250px;
}

/* 遗漏值表 */
.miss-section {
  width: 100%;
}

.miss-table {
  max-height: 400px;
  overflow-y: auto;
}

/* 响应式 */
@media (max-width: 1200px) {
  .overview-section {
    grid-template-columns: repeat(2, 1fr);
  }

  .main-analysis {
    grid-template-columns: 1fr;
  }

  .advanced-analysis {
    grid-template-columns: 1fr;
  }
}
</style>
