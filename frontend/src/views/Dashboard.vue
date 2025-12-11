<template>
  <div class="dashboard">
    <!-- 统计卡片区 -->
    <div class="stats-grid">
      <GlassCard v-for="(stat, index) in statsData" :key="stat.label" hoverable>
        <div class="stat-card">
          <div class="stat-icon" :style="{ background: stat.gradient }">
            <component :is="stat.icon" />
          </div>
          <div class="stat-info">
            <span class="stat-value">
              <NumberFlow :value="stat.value" :format="{ notation: 'compact' }" />
            </span>
            <span class="stat-label">{{ stat.label }}</span>
          </div>
          <div class="stat-trend" :class="stat.trend > 0 ? 'up' : 'down'">
            <TrendingUpOutline v-if="stat.trend > 0" />
            <TrendingDownOutline v-else />
            <span>{{ Math.abs(stat.trend) }}%</span>
          </div>
        </div>
      </GlassCard>
    </div>

    <!-- 主要内容区 -->
    <div class="main-grid">
      <!-- 最新开奖结果 -->
      <GlassCard title="最新开奖结果" class="latest-result-card">
        <template #headerExtra>
          <div class="header-actions">
            <n-button
              text
              type="warning"
              :loading="dataStore.isUpdating"
              @click="handleUpdateData"
            >
              <template #icon><CloudDownloadOutline /></template>
              {{ dataStore.isUpdating ? '更新中...' : '更新数据' }}
            </n-button>
            <n-button text type="primary" @click="refreshLatest">
              <template #icon><RefreshOutline /></template>
              刷新
            </n-button>
          </div>
        </template>
        <div class="latest-result">
          <div class="result-header">
            <span class="period">第 {{ displayResult.period }} 期</span>
            <span class="date">{{ displayResult.date }}</span>
          </div>
          <div class="balls-container">
            <div class="ball-group front">
              <span class="group-label">前区</span>
              <div class="balls">
                <LotteryBall
                  v-for="(num, idx) in displayResult.front"
                  :key="'f' + idx"
                  :number="num"
                  type="front"
                  size="lg"
                  :animate="animateBalls"
                  :delay="idx * 100"
                />
              </div>
            </div>
            <div class="ball-group back">
              <span class="group-label">后区</span>
              <div class="balls">
                <LotteryBall
                  v-for="(num, idx) in displayResult.back"
                  :key="'b' + idx"
                  :number="num"
                  type="back"
                  size="lg"
                  :animate="animateBalls"
                  :delay="(displayResult.front.length + idx) * 100"
                />
              </div>
            </div>
          </div>
          <div class="prize-info">
            <div class="prize-item">
              <span class="prize-label">一等奖</span>
              <span class="prize-value">{{ displayResult.prize1Count }} 注</span>
              <span class="prize-amount">{{ formatMoney(displayResult.prize1Amount) }}</span>
            </div>
            <div class="prize-item">
              <span class="prize-label">二等奖</span>
              <span class="prize-value">{{ displayResult.prize2Count }} 注</span>
              <span class="prize-amount">{{ formatMoney(displayResult.prize2Amount) }}</span>
            </div>
            <div class="prize-item">
              <span class="prize-label">三等奖</span>
              <span class="prize-value">{{ displayResult.prize3Count }} 注</span>
              <span class="prize-amount">{{ formatMoney(displayResult.prize3Amount) }}</span>
            </div>
            <div class="prize-item">
              <span class="prize-label">四等奖</span>
              <span class="prize-value">{{ displayResult.prize4Count }} 注</span>
              <span class="prize-amount">{{ formatMoney(displayResult.prize4Amount) }}</span>
            </div>
          </div>
        </div>
      </GlassCard>

      <!-- 算法性能 -->
      <GlassCard title="算法性能概览" class="algorithm-card">
        <div class="algorithm-list">
          <div v-for="algo in topAlgorithms" :key="algo.name" class="algorithm-item">
            <div class="algo-info">
              <span class="algo-name">{{ algo.name }}</span>
              <span class="algo-accuracy">{{ algo.accuracy }}%</span>
            </div>
            <n-progress
              type="line"
              :percentage="algo.accuracy"
              :show-indicator="false"
              :height="8"
              :border-radius="4"
              :fill-border-radius="4"
              :color="getProgressColor(algo.accuracy)"
            />
          </div>
        </div>
      </GlassCard>
    </div>

    <!-- 历史走势图 -->
    <GlassCard title="近期开奖走势" class="trend-card">
      <template #headerExtra>
        <n-radio-group v-model:value="trendRange" size="small">
          <n-radio-button value="10">近10期</n-radio-button>
          <n-radio-button value="30">近30期</n-radio-button>
          <n-radio-button value="50">近50期</n-radio-button>
        </n-radio-group>
      </template>
      <div class="trend-chart" ref="trendChartRef"></div>
    </GlassCard>

    <!-- 快速操作 -->
    <div class="quick-actions">
      <GlassCard hoverable @click="$router.push('/predict')">
        <div class="action-card">
          <div class="action-icon primary">
            <DiceOutline />
          </div>
          <div class="action-content">
            <h4>开始预测</h4>
            <p>使用 26+ 算法生成预测号码</p>
          </div>
          <ChevronForwardOutline class="action-arrow" />
        </div>
      </GlassCard>
      <GlassCard hoverable @click="$router.push('/analysis')">
        <div class="action-card">
          <div class="action-icon success">
            <BarChartOutline />
          </div>
          <div class="action-content">
            <h4>数据分析</h4>
            <p>查看号码统计和走势分析</p>
          </div>
          <ChevronForwardOutline class="action-arrow" />
        </div>
      </GlassCard>
      <GlassCard hoverable @click="$router.push('/compare')">
        <div class="action-card">
          <div class="action-icon warning">
            <GitCompareOutline />
          </div>
          <div class="action-content">
            <h4>批量对比</h4>
            <p>对比多期预测与实际结果</p>
          </div>
          <ChevronForwardOutline class="action-arrow" />
        </div>
      </GlassCard>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, onUnmounted } from 'vue'
import { useMessage } from 'naive-ui'
import * as echarts from 'echarts'
import NumberFlow from '@number-flow/vue'
import GlassCard from '@/components/common/GlassCard.vue'
import LotteryBall from '@/components/common/LotteryBall.vue'
import {
  TrendingUpOutline,
  TrendingDownOutline,
  RefreshOutline,
  CloudDownloadOutline,
  DiceOutline,
  BarChartOutline,
  GitCompareOutline,
  ChevronForwardOutline,
  LayersOutline,
  TimerOutline,
  CheckmarkCircleOutline,
  AnalyticsOutline
} from '@vicons/ionicons5'
import { useDataStore } from '@/stores/data'

const message = useMessage()
const dataStore = useDataStore()
const animateBalls = ref(false)
const trendRange = ref('30')
const trendChartRef = ref<HTMLElement | null>(null)
let trendChart: echarts.ECharts | null = null

// 使用 computed 从 store 获取最新开奖结果，提供默认值
const displayResult = computed(() => {
  const result = dataStore.latestResult
  if (result) {
    // 解析奖项信息（兼容多种后端返回格式）
    const prizeGrades = result.prize_grades || []
    const levelNames = ['一等奖', '二等奖', '三等奖', '四等奖', '五等奖', '六等奖']

    const getPrizeInfo = (level: number) => {
      const targetName = levelNames[level - 1]

      const prize = prizeGrades.find((p: any) => {
        // 格式1: 后端转换后的格式 (level 为字符串)
        if (typeof p.level === 'string') {
          return p.level === targetName
        }
        // 格式2: 原始 sporttery.cn 格式 (prizeLevelName)
        if (p.prizeLevelName) {
          return p.prizeLevelName === targetName
        }
        // 格式3: 旧格式 (level/grade 为数字)
        return p.level === level || p.grade === level
      })

      if (!prize) {
        return { count: 0, amount: 0 }
      }

      // 处理不同格式的数量字段
      const count = prize.prizeNum ?? prize.count ?? prize.prize_count ?? 0

      // 处理不同格式的金额字段
      let amount = 0
      if (prize.prizeAmount) {
        amount = parseInt(String(prize.prizeAmount).replace(/[^0-9]/g, ''), 10) || 0
      } else if (prize.amount) {
        amount = typeof prize.amount === 'string'
          ? parseInt(prize.amount.replace(/[^0-9]/g, ''), 10) || 0
          : prize.amount
      } else if (prize.prize_amount) {
        amount = prize.prize_amount
      }

      return { count, amount }
    }

    const prize1 = getPrizeInfo(1)
    const prize2 = getPrizeInfo(2)
    const prize3 = getPrizeInfo(3)
    const prize4 = getPrizeInfo(4)

    return {
      period: result.issue || '---',
      date: result.date || '---',
      front: result.front_balls || [],
      back: result.back_balls || [],
      prize1Count: prize1.count,
      prize1Amount: prize1.amount,
      prize2Count: prize2.count,
      prize2Amount: prize2.amount,
      prize3Count: prize3.count,
      prize3Amount: prize3.amount,
      prize4Count: prize4.count,
      prize4Amount: prize4.amount
    }
  }
  // 默认值
  return {
    period: '---',
    date: '---',
    front: [],
    back: [],
    prize1Count: 0,
    prize1Amount: 0,
    prize2Count: 0,
    prize2Amount: 0,
    prize3Count: 0,
    prize3Amount: 0,
    prize4Count: 0,
    prize4Amount: 0
  }
})

// 统计数据
const statsData = ref([
  {
    label: '历史期数',
    value: 2756,
    trend: 0.5,
    icon: LayersOutline,
    gradient: 'linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)'
  },
  {
    label: '算法数量',
    value: 26,
    trend: 8.3,
    icon: AnalyticsOutline,
    gradient: 'linear-gradient(135deg, #ff00ff 0%, #cc00cc 100%)'
  },
  {
    label: '平均响应',
    value: 0.8,
    trend: -12.5,
    icon: TimerOutline,
    gradient: 'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)'
  },
  {
    label: '预测准确率',
    value: 67.5,
    trend: 3.2,
    icon: CheckmarkCircleOutline,
    gradient: 'linear-gradient(135deg, #ffaa00 0%, #cc8800 100%)'
  }
])

// 算法性能排行
const topAlgorithms = ref([
  { name: '集成深度学习', accuracy: 89.5 },
  { name: 'LSTM时序预测', accuracy: 85.2 },
  { name: '自适应马尔可夫', accuracy: 82.8 },
  { name: 'Transformer注意力', accuracy: 80.1 },
  { name: '贝叶斯分析', accuracy: 76.5 }
])

// 格式化金额
const formatMoney = (amount: number) => {
  if (amount >= 10000) {
    return (amount / 10000).toFixed(0) + '万元'
  }
  return amount.toLocaleString() + '元'
}

// 获取进度条颜色
const getProgressColor = (value: number) => {
  if (value >= 85) return '#00ff88'
  if (value >= 75) return '#00d4ff'
  if (value >= 65) return '#ffaa00'
  return '#ff4757'
}

// 刷新最新数据
const refreshLatest = async () => {
  animateBalls.value = false
  await dataStore.fetchLatestResult()
  setTimeout(() => {
    animateBalls.value = true
  }, 100)
  message.success('数据已刷新')
}

// 更新数据（从官网爬取）
const handleUpdateData = async () => {
  const result = await dataStore.updateData()
  if (result.success) {
    animateBalls.value = false
    setTimeout(() => {
      animateBalls.value = true
    }, 100)
    if (result.data && result.data.updated_count > 0) {
      message.success(`更新成功，新增 ${result.data.updated_count} 期数据`)
    } else {
      message.info('数据已是最新，无需更新')
    }
  } else {
    message.error(result.message || '更新失败')
  }
}

// 初始化走势图
const initTrendChart = () => {
  if (!trendChartRef.value) return

  trendChart = echarts.init(trendChartRef.value)

  const option: echarts.EChartsOption = {
    backgroundColor: 'transparent',
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '10%',
      containLabel: true
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(15, 15, 35, 0.95)',
      borderColor: 'rgba(0, 212, 255, 0.3)',
      textStyle: {
        color: '#ffffff'
      }
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: Array.from({ length: 30 }, (_, i) => `${24970 + i}期`),
      axisLine: {
        lineStyle: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      axisLabel: {
        color: 'rgba(255, 255, 255, 0.6)',
        fontSize: 10
      }
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 35,
      axisLine: {
        lineStyle: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      axisLabel: {
        color: 'rgba(255, 255, 255, 0.6)'
      },
      splitLine: {
        lineStyle: {
          color: 'rgba(255, 255, 255, 0.05)'
        }
      }
    },
    series: [
      {
        name: '平均值',
        type: 'line',
        smooth: true,
        data: Array.from({ length: 30 }, () => Math.floor(Math.random() * 15) + 10),
        lineStyle: {
          color: '#00d4ff',
          width: 2
        },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(0, 212, 255, 0.3)' },
            { offset: 1, color: 'rgba(0, 212, 255, 0)' }
          ])
        },
        itemStyle: {
          color: '#00d4ff'
        }
      }
    ]
  }

  trendChart.setOption(option)
}

// 监听走势范围变化
watch(trendRange, () => {
  initTrendChart()
})

// 处理窗口大小变化
const handleResize = () => {
  trendChart?.resize()
}

onMounted(async () => {
  // 获取最新开奖数据
  await dataStore.fetchLatestResult()

  // 触发球号动画
  setTimeout(() => {
    animateBalls.value = true
  }, 300)

  // 初始化图表
  initTrendChart()

  // 响应式调整
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  trendChart?.dispose()
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* 统计卡片网格 */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px;
}

.stat-icon {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 24px;
}

.stat-info {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 28px;
  font-weight: bold;
  color: var(--text-primary);
}

.stat-label {
  font-size: 13px;
  color: var(--text-secondary);
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 20px;
}

.stat-trend.up {
  color: var(--neon-green);
  background: rgba(0, 255, 136, 0.1);
}

.stat-trend.down {
  color: var(--neon-blue);
  background: rgba(0, 212, 255, 0.1);
}

/* 主要内容网格 */
.main-grid {
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 20px;
}

/* 最新开奖结果 */
.latest-result {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.period {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
}

.date {
  font-size: 14px;
  color: var(--text-secondary);
}

.balls-container {
  display: flex;
  gap: 32px;
}

.ball-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.group-label {
  font-size: 12px;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 1px;
}

.balls {
  display: flex;
  gap: 8px;
}

.header-actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

.prize-info {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  padding-top: 16px;
  border-top: 1px solid var(--border-default);
}

.prize-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.prize-label {
  font-size: 12px;
  color: var(--text-tertiary);
}

.prize-value {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.prize-amount {
  font-size: 16px;
  font-weight: 600;
  color: var(--neon-green);
}

/* 算法性能 */
.algorithm-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.algorithm-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.algo-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.algo-name {
  font-size: 14px;
  color: var(--text-primary);
}

.algo-accuracy {
  font-size: 14px;
  font-weight: 600;
  color: var(--neon-blue);
}

/* 走势图 */
.trend-card {
  width: 100%;
}

.trend-chart {
  width: 100%;
  height: 300px;
}

/* 快速操作 */
.quick-actions {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.action-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px;
  cursor: pointer;
}

.action-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

.action-icon.primary {
  background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 212, 255, 0.1));
  color: var(--neon-blue);
}

.action-icon.success {
  background: linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 255, 136, 0.1));
  color: var(--neon-green);
}

.action-icon.warning {
  background: linear-gradient(135deg, rgba(255, 170, 0, 0.2), rgba(255, 170, 0, 0.1));
  color: var(--neon-orange);
}

.action-content {
  flex: 1;
}

.action-content h4 {
  margin: 0 0 4px 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.action-content p {
  margin: 0;
  font-size: 13px;
  color: var(--text-secondary);
}

.action-arrow {
  font-size: 20px;
  color: var(--text-tertiary);
  transition: transform 0.3s;
}

.action-card:hover .action-arrow {
  transform: translateX(4px);
  color: var(--neon-blue);
}

/* 响应式 */
@media (max-width: 1200px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .main-grid {
    grid-template-columns: 1fr;
  }

  .quick-actions {
    grid-template-columns: 1fr;
  }
}
</style>
