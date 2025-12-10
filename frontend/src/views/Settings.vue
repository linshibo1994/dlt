<template>
  <div class="settings-page">
    <!-- 基础设置 -->
    <GlassCard title="基础设置">
      <div class="settings-section">
        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">API 地址</span>
            <span class="setting-desc">后端 API 服务地址</span>
          </div>
          <n-input
            v-model:value="settings.apiBaseUrl"
            placeholder="http://localhost:8000"
            style="width: 300px"
          />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">默认分析期数</span>
            <span class="setting-desc">预测时默认使用的历史期数</span>
          </div>
          <n-input-number
            v-model:value="settings.defaultPeriods"
            :min="50"
            :max="500"
            style="width: 150px"
          />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">默认生成组数</span>
            <span class="setting-desc">每次预测默认生成的号码组数</span>
          </div>
          <n-input-number
            v-model:value="settings.defaultCount"
            :min="1"
            :max="100"
            style="width: 150px"
          />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">自动保存结果</span>
            <span class="setting-desc">预测完成后自动保存结果到本地</span>
          </div>
          <n-switch v-model:value="settings.autoSave" />
        </div>
      </div>
    </GlassCard>

    <!-- 界面设置 -->
    <GlassCard title="界面设置">
      <div class="settings-section">
        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">动画效果</span>
            <span class="setting-desc">启用界面动画和过渡效果</span>
          </div>
          <n-switch v-model:value="settings.enableAnimations" />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">号码球发光效果</span>
            <span class="setting-desc">预测结果号码球的发光动画</span>
          </div>
          <n-switch v-model:value="settings.ballGlow" />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">音效提示</span>
            <span class="setting-desc">预测完成时播放提示音</span>
          </div>
          <n-switch v-model:value="settings.soundEnabled" />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">紧凑模式</span>
            <span class="setting-desc">减少界面间距，显示更多内容</span>
          </div>
          <n-switch v-model:value="settings.compactMode" />
        </div>
      </div>
    </GlassCard>

    <!-- 算法设置 -->
    <GlassCard title="算法设置">
      <div class="settings-section">
        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">默认算法</span>
            <span class="setting-desc">预测时默认选中的算法</span>
          </div>
          <n-select
            v-model:value="settings.defaultAlgorithms"
            :options="algorithmOptions"
            multiple
            placeholder="选择默认算法"
            max-tag-count="responsive"
            style="width: 300px"
          />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">GPU 加速</span>
            <span class="setting-desc">使用 GPU 加速深度学习计算（需要 CUDA 支持）</span>
          </div>
          <n-switch v-model:value="settings.gpuAcceleration" />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">并行计算</span>
            <span class="setting-desc">多算法并行计算以提高速度</span>
          </div>
          <n-switch v-model:value="settings.parallelCompute" />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">置信度阈值</span>
            <span class="setting-desc">低于此阈值的结果将被标记</span>
          </div>
          <n-slider
            v-model:value="settings.confidenceThreshold"
            :min="30"
            :max="90"
            :step="5"
            :marks="{ 30: '30%', 60: '60%', 90: '90%' }"
            style="width: 300px"
          />
        </div>
      </div>
    </GlassCard>

    <!-- 数据设置 -->
    <GlassCard title="数据设置">
      <div class="settings-section">
        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">数据自动更新</span>
            <span class="setting-desc">启动时自动获取最新开奖数据</span>
          </div>
          <n-switch v-model:value="settings.autoUpdate" />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">更新频率</span>
            <span class="setting-desc">自动更新数据的间隔时间</span>
          </div>
          <n-select
            v-model:value="settings.updateInterval"
            :options="updateIntervalOptions"
            style="width: 150px"
          />
        </div>

        <div class="setting-item">
          <div class="setting-info">
            <span class="setting-label">缓存策略</span>
            <span class="setting-desc">预测结果的缓存保留时间</span>
          </div>
          <n-select
            v-model:value="settings.cacheStrategy"
            :options="cacheStrategyOptions"
            style="width: 150px"
          />
        </div>
      </div>
    </GlassCard>

    <!-- 系统信息 -->
    <GlassCard title="系统信息">
      <div class="system-info">
        <div class="info-grid">
          <div class="info-item">
            <span class="info-label">前端版本</span>
            <span class="info-value">v2.0.0</span>
          </div>
          <div class="info-item">
            <span class="info-label">后端版本</span>
            <span class="info-value">{{ systemInfo.backendVersion }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">数据期数</span>
            <span class="info-value">{{ systemInfo.totalPeriods }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">算法数量</span>
            <span class="info-value">{{ systemInfo.algorithmCount }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">最后更新</span>
            <span class="info-value">{{ systemInfo.lastUpdate }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">运行状态</span>
            <span class="info-value status-ok">正常</span>
          </div>
        </div>
      </div>
    </GlassCard>

    <!-- 操作按钮 -->
    <div class="settings-actions">
      <n-button type="primary" size="large" @click="saveSettings">
        <template #icon><SaveOutline /></template>
        保存设置
      </n-button>
      <n-button size="large" @click="resetSettings">
        <template #icon><RefreshOutline /></template>
        恢复默认
      </n-button>
      <n-button size="large" @click="clearCache">
        <template #icon><TrashOutline /></template>
        清除缓存
      </n-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useMessage, useDialog } from 'naive-ui'
import GlassCard from '@/components/common/GlassCard.vue'
import {
  SaveOutline,
  RefreshOutline,
  TrashOutline
} from '@vicons/ionicons5'
import { useSettingsStore } from '@/stores/settings'

const message = useMessage()
const dialog = useDialog()
const settingsStore = useSettingsStore()

// 设置数据 - 从 store 初始化
const settings = reactive({ ...settingsStore.settings })

// 算法选项
const algorithmOptions = [
  { label: '集成深度学习', value: 'ensemble' },
  { label: 'LSTM时序预测', value: 'lstm' },
  { label: '自适应马尔可夫', value: 'markov' },
  { label: 'Transformer', value: 'transformer' },
  { label: '贝叶斯分析', value: 'bayesian' },
  { label: '频率分析', value: 'frequency' }
]

// 更新间隔选项
const updateIntervalOptions = [
  { label: '30分钟', value: '30m' },
  { label: '1小时', value: '1h' },
  { label: '6小时', value: '6h' },
  { label: '12小时', value: '12h' },
  { label: '24小时', value: '24h' }
]

// 缓存策略选项
const cacheStrategyOptions = [
  { label: '1小时', value: '1h' },
  { label: '12小时', value: '12h' },
  { label: '24小时', value: '24h' },
  { label: '7天', value: '7d' },
  { label: '永久', value: 'forever' }
]

// 系统信息
const systemInfo = reactive({
  backendVersion: 'v1.5.0',
  totalPeriods: 2756,
  algorithmCount: 26,
  lastUpdate: '2025-01-01 10:00:00'
})

// 保存设置
const saveSettings = () => {
  settingsStore.updateSettings(settings)
  message.success('设置已保存')
}

// 重置设置
const resetSettings = () => {
  dialog.warning({
    title: '确认重置',
    content: '确定要恢复所有设置为默认值吗？',
    positiveText: '确定',
    negativeText: '取消',
    onPositiveClick: () => {
      settingsStore.resetSettings()
      // 同步到本地状态
      Object.assign(settings, settingsStore.settings)
      message.success('设置已恢复默认值')
    }
  })
}

// 清除缓存
const clearCache = () => {
  dialog.warning({
    title: '确认清除',
    content: '确定要清除所有缓存数据吗？这包括预测历史和临时数据。',
    positiveText: '确定',
    negativeText: '取消',
    onPositiveClick: () => {
      settingsStore.clearCache()
      message.success('缓存已清除')
    }
  })
}
</script>

<style scoped>
.settings-page {
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 900px;
}

/* 设置区块 */
.settings-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.setting-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid var(--border-default);
}

.setting-item:last-child {
  border-bottom: none;
}

.setting-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.setting-label {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.setting-desc {
  font-size: 12px;
  color: var(--text-tertiary);
}

/* 系统信息 */
.system-info {
  padding: 8px 0;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.info-label {
  font-size: 12px;
  color: var(--text-tertiary);
}

.info-value {
  font-size: 16px;
  font-weight: 500;
  color: var(--text-primary);
}

.info-value.status-ok {
  color: var(--neon-green);
}

/* 操作按钮 */
.settings-actions {
  display: flex;
  gap: 12px;
  padding-top: 16px;
}

/* 响应式 */
@media (max-width: 768px) {
  .setting-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .info-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .settings-actions {
    flex-direction: column;
  }
}
</style>
