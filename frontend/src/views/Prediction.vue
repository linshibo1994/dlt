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
        :disabled="selectedMethods.length === 0 || isPredicting"
        @click="startPredict"
      >
        <template #icon><FlashOutline /></template>
        {{ isPredicting ? '预测中...' : '开始预测' }}
      </n-button>
      <n-button
        v-if="isPredicting"
        type="error"
        size="large"
        @click="cancelPredict"
      >
        <template #icon><CloseCircleOutline /></template>
        取消预测
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
          <n-button text type="info" @click="copyAllResults">
            <template #icon><CopyOutline /></template>
            复制全部
          </n-button>
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
                  <div class="meta-item meta-action">
                    <n-button size="small" type="info" @click="copyResult(result)">
                      <template #icon><CopyOutline /></template>
                      复制号码
                    </n-button>
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
  CopyOutline,
  LayersOutline,
  AnalyticsOutline,
  GitNetworkOutline,
  HardwareChipOutline,
  PulseOutline,
  BulbOutline,
  TrendingUpOutline,
  StatsChartOutline,
  CubeOutline,
  ColorWandOutline,
  FlameOutline,
  SnowOutline,
  GridOutline,
  ExtensionPuzzleOutline,
  RocketOutline,
  DiamondOutline,
  ShuffleOutline,
  LinkOutline,
  CloseCircleOutline
} from '@vicons/ionicons5'
import { usePredictStore } from '@/stores/predict'
import { executePrediction } from '@/api/predict'
import type { PredictionRequest, PredictionResult } from '@/types'

const message = useMessage()
const predictStore = usePredictStore()

// 可用算法列表（共28种，按类别分组）
const availableMethods = ref([
  // 深度学习类
  {
    id: 'ensemble',
    name: '集成深度学习',
    description: '综合多种深度学习模型',
    category: 'deep_learning',
    icon: HardwareChipOutline,
    gradient: 'linear-gradient(135deg, #00d4ff 0%, #0099cc 100%)'
  },
  {
    id: 'lstm',
    name: 'LSTM时序预测',
    description: '长短期记忆网络',
    category: 'deep_learning',
    icon: PulseOutline,
    gradient: 'linear-gradient(135deg, #ff00ff 0%, #cc00cc 100%)'
  },
  {
    id: 'transformer',
    name: 'Transformer',
    description: '注意力机制模型',
    category: 'deep_learning',
    icon: LayersOutline,
    gradient: 'linear-gradient(135deg, #ffaa00 0%, #cc8800 100%)'
  },
  {
    id: 'gan',
    name: 'GAN生成对抗',
    description: '生成式对抗网络构造组合',
    category: 'deep_learning',
    icon: CubeOutline,
    gradient: 'linear-gradient(135deg, #ff6b6b 0%, #cc5555 100%)'
  },
  {
    id: 'stacking',
    name: 'Stacking集成',
    description: '深度模型堆叠融合',
    category: 'deep_learning',
    icon: ExtensionPuzzleOutline,
    gradient: 'linear-gradient(135deg, #4ecdc4 0%, #3da9a0 100%)'
  },
  {
    id: 'adaptive_ensemble',
    name: '自适应集成',
    description: '根据表现自适应调整',
    category: 'deep_learning',
    icon: ColorWandOutline,
    gradient: 'linear-gradient(135deg, #a29bfe 0%, #817fcb 100%)'
  },
  {
    id: 'ultimate_ensemble',
    name: '终极集成',
    description: '多通道终极加权',
    category: 'deep_learning',
    icon: DiamondOutline,
    gradient: 'linear-gradient(135deg, #fd79a8 0%, #ca6186 100%)'
  },

  // 传统统计类
  {
    id: 'frequency',
    name: '频率分析',
    description: '历史出现频率统计',
    category: 'traditional',
    icon: BulbOutline,
    gradient: 'linear-gradient(135deg, #9c88ff 0%, #7d6bcc 100%)'
  },
  {
    id: 'bayesian',
    name: '贝叶斯分析',
    description: '概率统计推断',
    category: 'traditional',
    icon: AnalyticsOutline,
    gradient: 'linear-gradient(135deg, #ff4757 0%, #cc3945 100%)'
  },
  {
    id: 'hot_cold',
    name: '冷热号分析',
    description: '结合热号冷号趋势',
    category: 'traditional',
    icon: FlameOutline,
    gradient: 'linear-gradient(135deg, #ff7675 0%, #cc5e5d 100%)'
  },
  {
    id: 'missing',
    name: '遗漏值分析',
    description: '评估号码遗漏周期',
    category: 'traditional',
    icon: SnowOutline,
    gradient: 'linear-gradient(135deg, #74b9ff 0%, #5c94cc 100%)'
  },

  // 马尔可夫链类
  {
    id: 'markov',
    name: '马尔可夫链',
    description: '一阶马尔可夫链分析',
    category: 'markov',
    icon: GitNetworkOutline,
    gradient: 'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)'
  },
  {
    id: 'markov_2nd',
    name: '二阶马尔可夫',
    description: '使用更长历史刻画惯性',
    category: 'markov',
    icon: LinkOutline,
    gradient: 'linear-gradient(135deg, #00b894 0%, #009276 100%)'
  },
  {
    id: 'markov_3rd',
    name: '三阶马尔可夫',
    description: '多阶链综合判断趋势',
    category: 'markov',
    icon: GitNetworkOutline,
    gradient: 'linear-gradient(135deg, #00cec9 0%, #00a4a1 100%)'
  },
  {
    id: 'adaptive_markov',
    name: '自适应马尔可夫',
    description: '动态调整阶数与权重',
    category: 'markov',
    icon: ColorWandOutline,
    gradient: 'linear-gradient(135deg, #55efc4 0%, #44bf9d 100%)'
  },
  {
    id: 'markov_custom',
    name: '自定义马尔可夫',
    description: '自定义分析期与预测期',
    category: 'markov',
    icon: LinkOutline,
    gradient: 'linear-gradient(135deg, #81ecec 0%, #67bdbd 100%)'
  },

  // 聚类算法类
  {
    id: 'clustering',
    name: '聚类预测',
    description: '向量化特征后进行K-Means',
    category: 'clustering',
    icon: GridOutline,
    gradient: 'linear-gradient(135deg, #6c5ce7 0%, #5646b9 100%)'
  },

  // 智能增强类
  {
    id: 'super',
    name: '超级预测',
    description: '调用超级预测器',
    category: 'intelligent',
    icon: RocketOutline,
    gradient: 'linear-gradient(135deg, #e17055 0%, #b45844 100%)'
  },
  {
    id: 'adaptive',
    name: '自适应学习',
    description: '多臂老虎机动态选择预测器',
    category: 'intelligent',
    icon: ColorWandOutline,
    gradient: 'linear-gradient(135deg, #fdcb6e 0%, #caa258 100%)'
  },
  {
    id: 'nine_models',
    name: '九模型融合',
    description: '九种数学模型投票',
    category: 'intelligent',
    icon: ExtensionPuzzleOutline,
    gradient: 'linear-gradient(135deg, #fab1a0 0%, #c88e80 100%)'
  },
  {
    id: 'advanced_integration',
    name: '高级集成',
    description: '综合热冷、马尔可夫、贝叶斯等',
    category: 'intelligent',
    icon: DiamondOutline,
    gradient: 'linear-gradient(135deg, #ff7675 0%, #cc5e5d 100%)'
  },
  {
    id: 'mixed_strategy',
    name: '混合策略',
    description: '多策略融合控制风险',
    category: 'intelligent',
    icon: ShuffleOutline,
    gradient: 'linear-gradient(135deg, #a29bfe 0%, #817fcb 100%)'
  },
  {
    id: 'highly_integrated',
    name: '高度集成',
    description: '多模型融合并加入评估',
    category: 'intelligent',
    icon: StatsChartOutline,
    gradient: 'linear-gradient(135deg, #74b9ff 0%, #5c94cc 100%)'
  },
  {
    id: 'enhanced',
    name: '增强引擎',
    description: '使用高级能力增强预测',
    category: 'intelligent',
    icon: RocketOutline,
    gradient: 'linear-gradient(135deg, #55efc4 0%, #44bf9d 100%)'
  },

  // 投注策略类
  {
    id: 'compound',
    name: '复式选号',
    description: '生成高覆盖复式组合',
    category: 'betting',
    icon: CopyOutline,
    gradient: 'linear-gradient(135deg, #fd79a8 0%, #ca6186 100%)'
  },
  {
    id: 'duplex',
    name: '胆拖选号',
    description: '根据胆码拖码构建组合',
    category: 'betting',
    icon: ExtensionPuzzleOutline,
    gradient: 'linear-gradient(135deg, #fdcb6e 0%, #caa258 100%)'
  },
  {
    id: 'markov_compound',
    name: '马尔可夫复式',
    description: '基于马尔可夫生成复式',
    category: 'betting',
    icon: LinkOutline,
    gradient: 'linear-gradient(135deg, #e17055 0%, #b45844 100%)'
  },
  {
    id: 'nine_models_compound',
    name: '九模型复式',
    description: '九模型结果生成复式',
    category: 'betting',
    icon: GridOutline,
    gradient: 'linear-gradient(135deg, #00b894 0%, #009276 100%)'
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
const abortController = ref<AbortController | null>(null)  // 用于取消预测

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

// 取消预测
const cancelPredict = () => {
  if (abortController.value) {
    abortController.value.abort()
    abortController.value = null
    isPredicting.value = false
    currentStep.value = ''
    message.info('预测已取消')

    // 重置步骤状态
    progressSteps.value.forEach(s => {
      s.completed = false
      s.active = false
    })
  }
}

// 开始预测
const startPredict = async () => {
  // 参数验证 - 双重保障（虽然UI已限制，但仍需编程式验证）
  if (selectedMethods.value.length === 0) {
    message.warning('请至少选择一种预测算法')
    return
  }

  if (params.value.count < 1 || params.value.count > 100) {
    message.warning('生成组数必须在 1-100 之间')
    return
  }

  if (params.value.periods < 50 || params.value.periods > 500) {
    message.warning('分析期数必须在 50-500 之间')
    return
  }

  if (params.value.betType === 'compound') {
    if (params.value.frontCount < 5 || params.value.frontCount > 10) {
      message.warning('复式投注前区个数必须在 5-10 之间')
      return
    }
    if (params.value.backCount < 2 || params.value.backCount > 5) {
      message.warning('复式投注后区个数必须在 2-5 之间')
      return
    }
  }

  // 创建 AbortController 用于取消请求
  abortController.value = new AbortController()

  isPredicting.value = true
  progress.value = 0
  predictionResults.value = []
  predictTime.value = 0

  const startTime = Date.now()

  try {
    // 步骤1: 数据加载
    progressSteps.value[0]!.active = true
    currentStep.value = '数据加载中...'
    progress.value = 10
    await new Promise(resolve => setTimeout(resolve, 300)) // 短暂延迟用于UI反馈
    progressSteps.value[0]!.completed = true
    progressSteps.value[0]!.active = false

    // 步骤2: 特征提取
    progressSteps.value[1]!.active = true
    currentStep.value = '特征提取中...'
    progress.value = 25

    // 计算每个算法应生成的组数（尽可能平均分配）
    const methodCount = selectedMethods.value.length
    const baseCountPerMethod = Math.floor(params.value.count / methodCount)
    const remainder = params.value.count % methodCount

    // 遍历所有选中的算法进行预测
    let totalProcessed = 0
    for (let i = 0; i < selectedMethods.value.length; i++) {
      const methodId = selectedMethods.value[i]

      // 计算当前算法应生成的组数（前面的算法分配余数）
      const countForMethod = i < remainder ? baseCountPerMethod + 1 : baseCountPerMethod

      if (countForMethod === 0) continue // 跳过不需要生成的算法

      // 获取算法信息
      const methodInfo = availableMethods.value.find(m => m.id === methodId)

      // 步骤3: 模型计算
      if (i === 0) {
        progressSteps.value[1]!.completed = true
        progressSteps.value[1]!.active = false
        progressSteps.value[2]!.active = true
      }
      currentStep.value = `正在使用 ${methodInfo?.name || methodId} 进行预测...`

      // 构建API请求参数
      const requestData: PredictionRequest = {
        method: methodId,
        periods: params.value.periods,
        count: countForMethod,
        front_count: params.value.frontCount,
        back_count: params.value.backCount,
        compound_mode: params.value.betType === 'compound'
      }

      // 调用后端API
      try {
        const response = await executePrediction(requestData)

        if (response.success && response.data) {
          // 根据投注模式处理不同的返回数据
          if (params.value.betType === 'compound' && response.data.compound) {
            // 复式投注结果：使用 compound 字段
            const compound = response.data.compound

            // 验证数据完整性
            if (Array.isArray(compound.front_balls) && Array.isArray(compound.back_balls)) {
              predictionResults.value.push({
                front: compound.front_balls,
                back: compound.back_balls,
                algorithm: methodInfo?.name || compound.method || methodId,
                confidence: typeof compound.confidence === 'number' ? compound.confidence : 65,
                timestamp: new Date().toLocaleTimeString()
              })
            } else {
              console.warn('复式预测结果数据不完整:', compound)
            }
          } else {
            // 单式投注结果：使用 predictions 数组
            const predictions = response.data.predictions || []

            predictions.forEach((pred: any) => {
              // 验证数据完整性
              if (!Array.isArray(pred.front_balls) || !Array.isArray(pred.back_balls)) {
                console.warn('预测结果数据不完整:', pred)
                return // 跳过无效数据
              }

              predictionResults.value.push({
                front: pred.front_balls,
                back: pred.back_balls,
                algorithm: methodInfo?.name || pred.method || methodId,
                confidence: typeof pred.confidence === 'number' ? pred.confidence : 65,
                timestamp: new Date().toLocaleTimeString()
              })
            })
          }
        } else {
          // API返回失败，使用降级处理
          message.warning(`算法 ${methodInfo?.name} 预测失败: ${response.message || '未知错误'}`)
        }
      } catch (error: any) {
        // 网络错误或其他异常
        console.error(`预测算法 ${methodId} 出错:`, error)
        message.error(`算法 ${methodInfo?.name} 调用失败: ${error.message || '网络错误'}`)
      }

      // 更新进度
      totalProcessed++
      const progressPercent = 25 + (totalProcessed / methodCount) * 60 // 25-85%
      progress.value = progressPercent
    }

    // 步骤4: 结果生成
    progressSteps.value[2]!.completed = true
    progressSteps.value[2]!.active = false
    progressSteps.value[3]!.active = true
    currentStep.value = '生成预测结果...'
    progress.value = 90

    await new Promise(resolve => setTimeout(resolve, 300))

    progressSteps.value[3]!.completed = true
    progressSteps.value[3]!.active = false
    progress.value = 100

    // 计算耗时
    predictTime.value = Number(((Date.now() - startTime) / 1000).toFixed(1))

    // 检查是否有成功的预测结果
    if (predictionResults.value.length > 0) {
      message.success(`预测完成！成功生成了 ${predictionResults.value.length} 组号码`)
    } else {
      message.error('预测失败，所有算法均未返回结果，请检查后端服务或更换算法')
    }

  } catch (error: any) {
    // 检查是否为取消操作
    if (error.name === 'AbortError' || error.name === 'CanceledError') {
      console.log('预测已被用户取消')
      return // 静默返回，不显示错误消息
    }
    console.error('预测过程出错:', error)
    message.error('预测过程发生错误: ' + (error.message || '未知错误'))
  } finally {
    isPredicting.value = false
    abortController.value = null  // 清理 AbortController

    // 重置步骤状态
    progressSteps.value.forEach(s => {
      s.completed = false
      s.active = false
    })
  }
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

// 复制文本到剪贴板（兼容多种浏览器）
const copyToClipboard = async (text: string): Promise<boolean> => {
  // 优先使用 Clipboard API
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch (err) {
      console.error('Clipboard API 失败:', err)
    }
  }
  // 降级方案：使用 execCommand
  const textArea = document.createElement('textarea')
  textArea.value = text
  textArea.style.position = 'fixed'
  textArea.style.left = '-9999px'
  textArea.style.top = '0'
  document.body.appendChild(textArea)
  textArea.focus()
  textArea.select()
  try {
    document.execCommand('copy')
    return true
  } catch (err) {
    console.error('execCommand 复制失败:', err)
    return false
  } finally {
    document.body.removeChild(textArea)
  }
}

// 复制单组号码
const copyResult = async (result: { front: number[]; back: number[] }) => {
  // 格式化号码：前区：01 02 03 04 05 | 后区：06 07
  const frontStr = result.front.map(n => n.toString().padStart(2, '0')).join(' ')
  const backStr = result.back.map(n => n.toString().padStart(2, '0')).join(' ')
  const text = `前区：${frontStr} | 后区：${backStr}`

  const success = await copyToClipboard(text)
  if (success) {
    message.success('号码已复制到剪贴板')
  } else {
    message.error('复制失败，请手动选择复制')
  }
}

// 复制全部号码
const copyAllResults = async () => {
  if (predictionResults.value.length === 0) {
    message.warning('暂无预测结果')
    return
  }

  const lines = predictionResults.value.map((result, idx) => {
    const frontStr = result.front.map(n => n.toString().padStart(2, '0')).join(' ')
    const backStr = result.back.map(n => n.toString().padStart(2, '0')).join(' ')
    return `第${idx + 1}组：前区 ${frontStr} | 后区 ${backStr}`
  })

  const success = await copyToClipboard(lines.join('\n'))
  if (success) {
    message.success(`已复制 ${predictionResults.value.length} 组号码`)
  } else {
    message.error('复制失败，请手动选择复制')
  }
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

.meta-action {
  justify-content: center;
  align-items: center;
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
