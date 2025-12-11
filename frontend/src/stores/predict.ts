import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { AlgorithmInfo, PredictionRequest, PredictionResult } from '@/types'
import { getAlgorithms, executePrediction } from '@/api/predict'

export const usePredictStore = defineStore('predict', () => {
  // 状态
  const methods = ref<AlgorithmInfo[]>([])
  const currentMethod = ref<string>('ensemble')
  const isLoading = ref(false)
  const progress = ref(0)
  const progressMessage = ref('')
  const results = ref<PredictionResult[]>([])

  // 获取预测方法列表
  const fetchMethods = async () => {
    try {
      const response = await getAlgorithms()
      if (response.success) {
        methods.value = response.data
      }
    } catch (error) {
      console.error('获取预测方法失败:', error)
    }
  }

  // 执行预测
  const predict = async (params: PredictionRequest) => {
    isLoading.value = true
    progress.value = 0
    progressMessage.value = '正在初始化...'

    // 模拟进度更新
    const progressSteps = ['数据加载中...', '特征提取中...', '模型计算中...', '结果生成中...']
    let step = 0
    let interval: ReturnType<typeof setInterval> | null = null

    try {
      interval = setInterval(() => {
        if (step < progressSteps.length) {
          progressMessage.value = progressSteps[step] ?? ''
          progress.value = (step + 1) * 25
          step++
        }
      }, 500)

      const response = await executePrediction(params)
      if (response.success && response.data) {
        // response.data 是 PredictionResponseData，提取其中的 predictions
        results.value = response.data.predictions || []
      }

      progress.value = 100
      progressMessage.value = '预测完成!'
    } catch (error) {
      console.error('预测失败:', error)
      progressMessage.value = '预测失败'
    } finally {
      // 确保 interval 被清除，防止内存泄漏
      if (interval) {
        clearInterval(interval)
      }
      isLoading.value = false
    }
  }

  // 清空结果
  const clearResults = () => {
    results.value = []
    progress.value = 0
    progressMessage.value = ''
  }

  return {
    methods,
    currentMethod,
    isLoading,
    progress,
    progressMessage,
    results,
    fetchMethods,
    predict,
    clearResults
  }
})