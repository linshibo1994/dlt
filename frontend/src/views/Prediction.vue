<template>
  <div class="prediction-page">
    <!-- 配置面板 -->
    <div class="config-section">
      <!-- 算法选择 -->
      <GlassCard title="选择预测算法">
        <template #headerExtra>
          <n-button text size="small" @click="selectAllMethods">全选</n-button>
          <n-button text size="small" @click="clearMethods">清除</n-button>
        </template>
        <div class="method-grid">
          <div
            v-for="method in availableMethods"
            :key="method.id"
            class="method-card"
            :class="{ selected: selectedMethods.includes(method.id) }"
            @click="toggleMethod(method.id)"
          >
            <div class="method-icon" :style="{ background: method.gradient }">
              <component :is="method.icon" />
            </div>
            <div class="method-info">
              <span class="method-name">{{ method.name }}</span>
              <span class="method-desc">{{ method.description }}</span>
            </div>
            <div class="method-check">
              <CheckmarkCircleOutline v-if="selectedMethods.includes(method.id)" />
              <EllipseOutline v-else />
            </div>
          </div>
        </div>
      </GlassCard>

      <!-- 参数配置 -->
      <GlassCard title="预测参数">
        <div class="params-form">
          <div class="param-item">
            <label>分析期数</label>
            <n-slider
              v-model:value="params.periods"
              :min="50"
              :max="500"
              :step="10"
              :marks="periodMarks"
            />
            <span class="param-value">{{ params.periods }} 期</span>
          </div>

          <div class="param-item">
            <label>生成组数</label>
            <n-input-number
              v-model:value="params.count"
              :min="1"
              :max="100"
              placeholder="生成号码组数"
            />
          </div>

          <div class="param-item">
            <label>投注方式</label>
            <n-select
              v-model:value="params.betType"
              :options="betTypeOptions"
              placeholder="选择投注方式"
            />
          </div>

          <div v-if="params.betType === 'compound'" class="param-item">
            <label>复式配置</label>
            <div class="compound-config">
              <n-input-number
                v-model:value="params.frontCount"
                :min="5"
                :max="10"
                placeholder="前区个数"
              >
                <template #prefix>前区</template>
              </n-input-number>
              <n-input-number
                v-model:value="params.backCount"
                :min="2"
                :max="5"
                placeholder="后区个数"
              >
                <template #prefix>后区</template>
              </n-input-number>
            </div>
          </div>
        </div>
      </GlassCard>
    </div>

    <!-- 预测按钮 -->
    <div class="predict-action">
      <n-button
        type="primary"
        size="large"
        :loading="isPredicting"
        :disabled="selectedMethods.length === 0"
        @click="startPredict"
      >
        <template #icon><FlashOutline /></template>
        {{ isPredicting ? '预测中...' : '开始预测' }}
      </n-button>
      <span class="action-hint">
        已选择 {{ selectedMethods.length }} 种算法，将生成 {{ params.count }} 组号码
      </span>
    </div>

    <!-- 预测进度 -->
    <GlassCard v-if="isPredicting" title="预测进度" glow>
      <div class="progress-section">
        <div class="progress-visual">
          <ProgressRing :value="progress" :size="160" :stroke-width="10" />
        </div>
        <div class="progress-info">
          <div class="progress-status">
            <LoadingSpinner v-if="isPredicting" size="sm" />
            <span class="status-text">{{ currentStep }}</span>
          </div>
          <div class="progress-details">
            <div v-for="(step, idx) in progressSteps" :key="idx" class="step-item">
              <CheckmarkCircleOutline v-if="step.completed" class="step-done" />
              <EllipseOutline v-else-if="step.active" class="step-active" />
              <EllipseOutline v-else class="step-pending" />
              <span :class="{ active: step.active, done: step.completed }">{{ step.name }}</span>
            </div>
          </div>
        </div>
      </div>
    </GlassCard>

    <!-- 预测结果 -->
    <div v-if="predictionResults.length > 0" class="results-section">
      <GlassCard title="预测结果" class="results-card">
        <template #headerExtra>
          <n-button text type="primary" @click="exportResults">
            <template #icon><DownloadOutline /></template>
            导出
          </n-button>
          <n-button text type="error" @click="clearResults">
            <template #icon><TrashOutline /></template>
            清除
          </n-button>
        </template>

        <div class="results-container">
          <n-tabs type="line" animated>
            <n-tab-pane
              v-for="(result, idx) in predictionResults"
              :key="idx"
              :name="'result-' + idx"
              :tab="'第 ' + (idx + 1) + ' 组'"
            >
              <div class="result-item">
                <div class="result-balls">
                  <div class="ball-section">
                    <span class="section-label">前区号码</span>
                    <div class="balls-row">
                      <LotteryBall
                        v-for="(num, ballIdx) in result.front"
                        :key="'f-' + ballIdx"
                        :number="num"
                        type="front"
                        size="xl"
                        animate
                        glow
                        :delay="ballIdx * 150"
                      />
                    </div>
                  </div>
                  <div class="ball-section">
                    <span class="section-label">后区号码</span>
                    <div class="balls-row">
                      <LotteryBall
                        v-for="(num, ballIdx) in result.back"
                        :key="'b-' + ballIdx"
                        :number="num"
                        type="back"
                        size="xl"
                        animate
                        glow
                        :delay="(result.front.length + ballIdx) * 150"
                      />
                    </div>
                  </div>
                </div>

                <div class="result-meta">
                  <div class="meta-item">
                    <span class="meta-label">推荐算法</span>
                    <span class="meta-value">{{ result.algorithm }}</span>
                  </div>
                  <div class="meta-item">
                    <span class="meta-label">置信度</span>
                    <n-progress
                      type="line"
                      :percentage="result.confidence"
                      :height="8"
                      :border-radius="4"
                      :color="getConfidenceColor(result.confidence)"
                    />
                  </div>
                  <div class="meta-item">
                    <span class="meta-label">生成时间</span>
                    <span class="meta-value">{{ result.timestamp }}</span>
                  </div>
                </div>
              </div>
            </n-tab-pane>
          </n-tabs>
        </div>
      </GlassCard>

      <!-- 结果统计 -->
      <div class="results-stats">
        <GlassCard hoverable>
          <div class="stat-mini">
            <span class="stat-mini-value">{{ predictionResults.length }}</span>
            <span class="stat-mini-label">生成组数</span>
          </div>
        </GlassCard>
        <GlassCard hoverable>
          <div class="stat-mini">
            <span class="stat-mini-value">{{ averageConfidence }}%</span>
            <span class="stat-mini-label">平均置信度</span>
          </div>
        </GlassCard>
        <GlassCard hoverable>
          <div class="stat-mini">
            <span class="stat-mini-value">{{ selectedMethods.length }}</span>
            <span class="stat-mini-label">使用算法</span>
          </div>
        </GlassCard>
        <GlassCard hoverable>
          <div class="stat-mini">
            <span class="stat-mini-value">{{ predictTime }}s</span>
            <span class="stat-mini-label">耗时</span>
          </div>
        </GlassCard>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useMessage } from 'naive-ui'
import GlassCard from '@/components/common/GlassCard.vue'
import LotteryBall from '@/components/common/LotteryBall.vue'
import ProgressRing from '@/components/common/ProgressRing.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import {
  FlashOutline,
  CheckmarkCircleOutline,
  EllipseOutline,
  DownloadOutline,
  TrashOutline,
  LayersOutline,
  AnalyticsOutline,
  GitNetworkOutline,
  HardwareChipOutline,
  PulseOutline,
  BulbOutline
} from '@vicons/ionicons5'
import { usePredictStore } from '@/stores/predict'

const message = useMessage()
const predictStore = usePredictStore()

// 可用算法列表
const availableMethods = ref([
  {
    id: 'ensemble',
    name: '集成深度学习',
    description: '综合多种深度学习模型',
    icon: HardwareChipOutline,
    gradient: 'linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)'
  },
  {
    id: 'lstm',
    name: 'LSTM时序预测',
    description: '长短期记忆网络',
    icon: PulseOutline,
    gradient: 'linear-gradient(135deg, #ff00ff 0%, #cc00cc 100%)'
  },
  {
    id: 'markov',
    name: '自适应马尔可夫',
    description: '动态马尔可夫链分析',
    icon: GitNetworkOutline,
    gradient: 'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)'
  },
  {
    id: 'transformer',
    name: 'Transformer',
    description: '注意力机制模型',
    icon: LayersOutline,
    gradient: 'linear-gradient(135deg, #ffaa00 0%, #cc8800 100%)'
  },
  {
    id: 'bayesian',
    name: '贝叶斯分析',
    description: '概率统计推断',
    icon: AnalyticsOutline,
    gradient: 'linear-gradient(135deg, #ff4757 0%, #cc3945 100%)'
  },
  {
    id: 'frequency',
    name: '频率分析',
    description: '历史出现频率统计',
    icon: BulbOutline,
    gradient: 'linear-gradient(135deg, #9c88ff 0%, #7d6bcc 100%)'
  }
])

// 选中的算法
const selectedMethods = ref<string[]>(['ensemble', 'lstm'])

// 参数配置
const params = ref({
  periods: 200,
  count: 5,
  betType: 'single',
  frontCount: 6,
  backCount: 3
})

// 期数标记
const periodMarks = {
  50: '50',
  100: '100',
  200: '200',
  300: '300',
  500: '500'
}

// 投注方式选项
const betTypeOptions = [
  { label: '单式投注', value: 'single' },
  { label: '复式投注', value: 'compound' },
  { label: '胆拖投注', value: 'dantuo' }
]

// 预测状态
const isPredicting = ref(false)
const progress = ref(0)
const currentStep = ref('')
const predictTime = ref(0)

// 进度步骤
const progressSteps = ref([
  { name: '数据加载', completed: false, active: false },
  { name: '特征提取', completed: false, active: false },
  { name: '模型计算', completed: false, active: false },
  { name: '结果生成', completed: false, active: false }
])

// 预测结果
const predictionResults = ref<Array<{
  front: number[]
  back: number[]
  algorithm: string
  confidence: number
  timestamp: string
}>>([])

// 平均置信度
const averageConfidence = computed(() => {
  if (predictionResults.value.length === 0) return 0
  const sum = predictionResults.value.reduce((acc, r) => acc + r.confidence, 0)
  return Math.round(sum / predictionResults.value.length)
})

// 切换算法选择
const toggleMethod = (id: string) => {
  const index = selectedMethods.value.indexOf(id)
  if (index === -1) {
    selectedMethods.value.push(id)
  } else {
    selectedMethods.value.splice(index, 1)
  }
}

// 全选算法
const selectAllMethods = () => {
  selectedMethods.value = availableMethods.value.map(m => m.id)
}

// 清除选择
const clearMethods = () => {
  selectedMethods.value = []
}

// 开始预测
const startPredict = async () => {
  isPredicting.value = true
  progress.value = 0
  predictionResults.value = []
  predictTime.value = 0

  const startTime = Date.now()

  // 模拟预测进度
  const steps = progressSteps.value
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i]
    if (step) {
      step.active = true
      currentStep.value = step.name + '中...'

      await simulateStep(i)

      step.completed = true
      step.active = false
    }
    progress.value = ((i + 1) / steps.length) * 100
  }

  // 生成模拟结果
  for (let i = 0; i < params.value.count; i++) {
    const front = generateRandomNumbers(5, 1, 35)
    const back = generateRandomNumbers(2, 1, 12)
    const method = availableMethods.value.find(
      m => m.id === selectedMethods.value[i % selectedMethods.value.length]
    )

    predictionResults.value.push({
      front,
      back,
      algorithm: method?.name || '集成预测',
      confidence: Math.floor(Math.random() * 30) + 60,
      timestamp: new Date().toLocaleTimeString()
    })
  }

  predictTime.value = ((Date.now() - startTime) / 1000).toFixed(1) as unknown as number
  isPredicting.value = false
  message.success(`预测完成！生成了 ${params.value.count} 组号码`)

  // 重置步骤状态
  progressSteps.value.forEach(s => {
    s.completed = false
    s.active = false
  })
}

// 模拟步骤延迟
const simulateStep = (stepIndex: number): Promise<void> => {
  const delays = [500, 800, 1200, 600]
  return new Promise(resolve => setTimeout(resolve, delays[stepIndex]))
}

// 生成随机号码
const generateRandomNumbers = (count: number, min: number, max: number): number[] => {
  const numbers: number[] = []
  while (numbers.length < count) {
    const num = Math.floor(Math.random() * (max - min + 1)) + min
    if (!numbers.includes(num)) {
      numbers.push(num)
    }
  }
  return numbers.sort((a, b) => a - b)
}

// 获取置信度颜色
const getConfidenceColor = (value: number) => {
  if (value >= 80) return '#00ff88'
  if (value >= 60) return '#00d4ff'
  if (value >= 40) return '#ffaa00'
  return '#ff4757'
}

// 导出结果
const exportResults = () => {
  const data = predictionResults.value.map((r, i) => ({
    序号: i + 1,
    前区: r.front.join(','),
    后区: r.back.join(','),
    算法: r.algorithm,
    置信度: r.confidence + '%',
    时间: r.timestamp
  }))

  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `prediction_${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)
  message.success('导出成功')
}

// 清除结果
const clearResults = () => {
  predictionResults.value = []
  message.info('结果已清除')
}
</script>

<style scoped>
.prediction-page {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* 配置区域 */
.config-section {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;
}

/* 算法选择网格 */
.method-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.method-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-default);
  background: rgba(0, 0, 0, 0.2);
  cursor: pointer;
  transition: all var(--transition-fast);
}

.method-card:hover {
  border-color: var(--border-glow);
  background: rgba(0, 212, 255, 0.05);
}

.method-card.selected {
  border-color: var(--neon-blue);
  background: rgba(0, 212, 255, 0.1);
}

.method-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 20px;
  flex-shrink: 0;
}

.method-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.method-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.method-desc {
  font-size: 11px;
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.method-check {
  font-size: 20px;
  color: var(--text-tertiary);
}

.method-card.selected .method-check {
  color: var(--neon-blue);
}

/* 参数表单 */
.params-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.param-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.param-item label {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
}

.param-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--neon-blue);
  text-align: right;
}

.compound-config {
  display: flex;
  gap: 12px;
}

/* 预测按钮 */
.predict-action {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 24px;
}

.predict-action .n-button {
  min-width: 200px;
  height: 48px;
  font-size: 16px;
}

.action-hint {
  font-size: 13px;
  color: var(--text-tertiary);
}

/* 进度区域 */
.progress-section {
  display: flex;
  align-items: center;
  gap: 40px;
  padding: 20px;
}

.progress-visual {
  flex-shrink: 0;
}

.progress-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.progress-status {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-text {
  font-size: 18px;
  font-weight: 500;
  color: var(--text-primary);
}

.progress-details {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.step-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: var(--text-tertiary);
}

.step-done {
  color: var(--neon-green);
}

.step-active {
  color: var(--neon-blue);
  animation: pulse 1s ease-in-out infinite;
}

.step-pending {
  color: var(--text-tertiary);
}

.step-item span.active {
  color: var(--neon-blue);
}

.step-item span.done {
  color: var(--neon-green);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* 结果区域 */
.results-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.results-container {
  padding: 12px 0;
}

.result-item {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 20px 0;
}

.result-balls {
  display: flex;
  gap: 48px;
}

.ball-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.section-label {
  font-size: 14px;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 2px;
}

.balls-row {
  display: flex;
  gap: 12px;
}

.result-meta {
  display: flex;
  gap: 32px;
  padding-top: 20px;
  border-top: 1px solid var(--border-default);
}

.meta-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 150px;
}

.meta-label {
  font-size: 12px;
  color: var(--text-tertiary);
}

.meta-value {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

/* 结果统计 */
.results-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.stat-mini {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 16px;
}

.stat-mini-value {
  font-size: 28px;
  font-weight: bold;
  color: var(--neon-blue);
}

.stat-mini-label {
  font-size: 12px;
  color: var(--text-tertiary);
}

/* 响应式 */
@media (max-width: 1200px) {
  .config-section {
    grid-template-columns: 1fr;
  }

  .method-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .results-stats {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 768px) {
  .method-grid {
    grid-template-columns: 1fr;
  }

  .result-balls {
    flex-direction: column;
    gap: 24px;
  }

  .result-meta {
    flex-direction: column;
    gap: 16px;
  }

  .results-stats {
    grid-template-columns: 1fr;
  }
}
</style>
