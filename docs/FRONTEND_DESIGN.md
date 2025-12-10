# 大乐透智能预测系统 - 前端设计文档

## 1. 设计概述

### 1.1 设计风格
**Cyberpunk + Glassmorphism (赛博朋克 + 玻璃拟态)**

结合科技感的霓虹灯效果与现代玻璃拟态设计，打造沉浸式的彩票预测体验。

### 1.2 技术栈

| 类别 | 技术 | 版本 | 说明 |
|------|------|------|------|
| 框架 | Vue | 3.5+ | Composition API + script setup |
| 语言 | TypeScript | 5.0+ | 类型安全 |
| 构建 | Vite | 6.0+ | 极速构建 |
| UI库 | Naive UI | 2.38+ | 80+组件，暗色主题 |
| 样式 | Tailwind CSS | 4.0+ | 原子化CSS |
| 图表 | Apache ECharts | 5.5+ | 数据可视化 |
| 动画 | GSAP | 3.12+ | 高性能动画 |
| 数字动效 | @number-flow/vue | latest | 数字滚动 |
| 状态管理 | Pinia | 2.1+ | Vue官方推荐 |
| 路由 | Vue Router | 4.3+ | SPA路由 |
| HTTP | Axios | 1.6+ | API请求 |

## 2. 色彩系统

### 2.1 主色调

```css
:root {
  /* 背景色 */
  --bg-primary: #0a0a1a;
  --bg-secondary: #1a1a3e;
  --bg-card: rgba(26, 26, 62, 0.6);

  /* 霓虹色 */
  --neon-blue: #00d4ff;
  --neon-blue-dark: #0099ff;
  --neon-purple: #ff00ff;
  --neon-purple-dark: #9d00ff;
  --neon-green: #00ff88;
  --neon-yellow: #ffff00;
  --neon-red: #ff3366;

  /* 前区球色 (霓虹蓝) */
  --ball-front-primary: #00d4ff;
  --ball-front-secondary: #0099ff;
  --ball-front-glow: 0 0 20px rgba(0, 212, 255, 0.6);

  /* 后区球色 (霓虹紫) */
  --ball-back-primary: #ff00ff;
  --ball-back-secondary: #9d00ff;
  --ball-back-glow: 0 0 20px rgba(255, 0, 255, 0.6);

  /* 文字色 */
  --text-primary: #ffffff;
  --text-secondary: rgba(255, 255, 255, 0.7);
  --text-muted: rgba(255, 255, 255, 0.4);

  /* 边框色 */
  --border-glow: rgba(0, 212, 255, 0.3);
  --border-default: rgba(255, 255, 255, 0.1);
}
```

### 2.2 渐变效果

```css
/* 主背景渐变 */
.bg-gradient-main {
  background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0a0a1a 100%);
}

/* 卡片背景 */
.glass-card {
  background: rgba(26, 26, 62, 0.6);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(0, 212, 255, 0.2);
  box-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
}

/* 霓虹边框 */
.neon-border {
  border: 1px solid transparent;
  background: linear-gradient(#1a1a3e, #1a1a3e) padding-box,
              linear-gradient(135deg, #00d4ff, #ff00ff) border-box;
}
```

## 3. 页面结构

### 3.1 整体布局

```
+----------------------------------------------------------+
|                      Header                               |
+----------+-----------------------------------------------+
|          |                                               |
|          |                                               |
|  Sidebar |              Main Content                     |
|          |                                               |
|          |                                               |
+----------+-----------------------------------------------+
```

### 3.2 页面导航

| 页面 | 路由 | 图标 | 说明 |
|------|------|------|------|
| Dashboard | / | HomeOutline | 系统概览 |
| Prediction | /predict | DiceOutline | 号码预测 |
| Analysis | /analysis | BarChartOutline | 数据分析 |
| Compare | /compare | GitCompareOutline | 批量对比 |
| Settings | /settings | SettingsOutline | 系统设置 |

## 4. 核心组件设计

### 4.1 GlassCard 玻璃卡片

```vue
<template>
  <div class="glass-card" :class="{ 'glow': glow }">
    <div v-if="title" class="card-header">
      <h3 class="card-title">{{ title }}</h3>
      <slot name="header-extra" />
    </div>
    <div class="card-content">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.glass-card {
  background: rgba(26, 26, 62, 0.6);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  border: 1px solid rgba(0, 212, 255, 0.2);
  padding: 24px;
  transition: all 0.3s ease;
}

.glass-card:hover {
  border-color: rgba(0, 212, 255, 0.4);
  box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
}

.glass-card.glow {
  box-shadow: 0 0 40px rgba(0, 212, 255, 0.2);
}
</style>
```

### 4.2 LotteryBall 彩球组件

```vue
<template>
  <div
    class="lottery-ball"
    :class="[type, { 'animate': animate }]"
  >
    <span class="ball-number">{{ number }}</span>
  </div>
</template>

<style scoped>
.lottery-ball {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 18px;
  transition: all 0.3s ease;
}

/* 前区球 - 霓虹蓝 */
.lottery-ball.front {
  background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
  box-shadow: 0 0 20px rgba(0, 212, 255, 0.6),
              inset 0 -3px 10px rgba(0, 0, 0, 0.2);
  color: #0a0a1a;
}

/* 后区球 - 霓虹紫 */
.lottery-ball.back {
  background: linear-gradient(135deg, #ff00ff 0%, #9d00ff 100%);
  box-shadow: 0 0 20px rgba(255, 0, 255, 0.6),
              inset 0 -3px 10px rgba(0, 0, 0, 0.2);
  color: #0a0a1a;
}

/* 入场动画 */
.lottery-ball.animate {
  animation: ballPop 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

@keyframes ballPop {
  0% { transform: scale(0) rotate(-180deg); opacity: 0; }
  50% { transform: scale(1.2) rotate(0deg); }
  100% { transform: scale(1) rotate(0deg); opacity: 1; }
}

/* 悬停效果 */
.lottery-ball:hover {
  transform: scale(1.1);
  box-shadow: 0 0 30px rgba(0, 212, 255, 0.8);
}
</style>
```

### 4.3 PredictionProgress 预测进度组件

```vue
<template>
  <div class="prediction-progress">
    <!-- 进度环 -->
    <div class="progress-ring">
      <svg viewBox="0 0 100 100">
        <circle class="progress-bg" cx="50" cy="50" r="45" />
        <circle
          class="progress-bar"
          cx="50" cy="50" r="45"
          :style="{ strokeDashoffset: dashOffset }"
        />
      </svg>
      <div class="progress-text">
        <NumberFlow :value="percentage" />%
      </div>
    </div>

    <!-- 步骤列表 -->
    <div class="steps">
      <div
        v-for="(step, index) in steps"
        :key="index"
        class="step"
        :class="{ 'active': index === currentStep, 'completed': index < currentStep }"
      >
        <div class="step-indicator">
          <div v-if="index < currentStep" class="check-icon">check</div>
          <div v-else class="step-number">{{ index + 1 }}</div>
        </div>
        <span class="step-label">{{ step }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.progress-ring {
  width: 120px;
  height: 120px;
  position: relative;
}

.progress-ring svg {
  transform: rotate(-90deg);
}

.progress-bg {
  fill: none;
  stroke: rgba(0, 212, 255, 0.1);
  stroke-width: 8;
}

.progress-bar {
  fill: none;
  stroke: url(#gradient);
  stroke-width: 8;
  stroke-linecap: round;
  stroke-dasharray: 283;
  transition: stroke-dashoffset 0.5s ease;
}

.progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 24px;
  font-weight: bold;
  color: #00d4ff;
  text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
}

.step.active .step-indicator {
  background: linear-gradient(135deg, #00d4ff, #ff00ff);
  box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
}

.step.completed .step-indicator {
  background: #00ff88;
}
</style>
```

### 4.4 AnimatedNumber 数字动效

```vue
<template>
  <NumberFlow
    :value="value"
    :format="format"
    class="animated-number"
  />
</template>

<script setup lang="ts">
import NumberFlow from '@number-flow/vue'

defineProps<{
  value: number
  format?: Intl.NumberFormatOptions
}>()
</script>

<style scoped>
.animated-number {
  font-variant-numeric: tabular-nums;
  color: #00d4ff;
  text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
}
</style>
```

## 5. 页面设计

### 5.1 Dashboard 首页

**布局结构**：
```
+------------------+------------------+------------------+
|   系统状态卡片   |   最新开奖结果   |    快速预测     |
+------------------+------------------+------------------+
|                                                       |
|              历史数据走势图 (ECharts)                  |
|                                                       |
+---------------------------+---------------------------+
|     号码频率分布          |      冷热号分析           |
+---------------------------+---------------------------+
```

**关键功能**：
- 实时系统状态显示（CPU/内存/GPU使用率）
- 最新开奖结果展示（带动效）
- 一键快速预测入口
- 历史数据走势图
- 号码频率分布热力图

### 5.2 Prediction 预测页面 (核心)

**布局结构**：
```
+-------------------------------------------------------+
|                    方法选择区域                        |
|  [频率分析] [马尔可夫] [LSTM] [集成预测] [更多...]    |
+-------------------------------------------------------+
|                                                       |
|  +-------------------+  +---------------------------+ |
|  |                   |  |                           | |
|  |   参数配置面板    |  |      预测结果展示区       | |
|  |                   |  |                           | |
|  |  - 分析期数       |  |   前区: 01 05 12 23 35   | |
|  |  - 生成注数       |  |   后区: 03 08            | |
|  |  - 加速模式       |  |                           | |
|  |  - 复式选项       |  |   置信度: 0.85           | |
|  |                   |  |                           | |
|  +-------------------+  +---------------------------+ |
|                                                       |
+-------------------------------------------------------+
|                    预测进度动画                        |
|  [数据加载] -> [特征提取] -> [模型计算] -> [结果生成]  |
+-------------------------------------------------------+
```

**动效设计**：
1. **方法选择动效**：悬停时卡片上浮 + 边框发光
2. **参数调整动效**：滑块拖动时数值实时变化
3. **预测进度动效**：
   - 环形进度条渐变填充
   - 步骤指示器依次点亮
   - 当前步骤脉冲动画
4. **结果展示动效**：
   - 号码球依次弹出（stagger动画）
   - 数字滚动显示置信度
   - 背景粒子效果

### 5.3 Analysis 数据分析页面

**布局结构**：
```
+-------------------------------------------------------+
|                    分析维度选择                        |
|  [频率分析] [冷热分析] [遗漏分析] [走势分析]          |
+-------------------------------------------------------+
|                                                       |
|  +-------------------------+  +---------------------+ |
|  |                         |  |                     | |
|  |    主图表区域           |  |   数据统计面板      | |
|  |    (ECharts)            |  |                     | |
|  |                         |  |   - 出现次数        | |
|  |                         |  |   - 平均遗漏        | |
|  |                         |  |   - 最大遗漏        | |
|  |                         |  |                     | |
|  +-------------------------+  +---------------------+ |
|                                                       |
+-------------------------------------------------------+
|                    号码详情表格                        |
+-------------------------------------------------------+
```

**图表类型**：
- 柱状图：号码出现频率
- 热力图：号码冷热分布
- 折线图：遗漏值走势
- 散点图：号码关联分析

### 5.4 Compare 批量对比页面

**布局结构**：
```
+-------------------------------------------------------+
|                    对比配置面板                        |
|  期号输入  预测方法  分析期数  对比次数  [开始对比]   |
+-------------------------------------------------------+
|                                                       |
|  +-------------------------+  +---------------------+ |
|  |                         |  |                     | |
|  |    对比结果统计         |  |   中奖等级分布      | |
|  |                         |  |   (饼图/环形图)     | |
|  |    总对比次数: 100      |  |                     | |
|  |    中奖次数: 23         |  |                     | |
|  |    中奖率: 23%          |  |                     | |
|  |                         |  |                     | |
|  +-------------------------+  +---------------------+ |
|                                                       |
+-------------------------------------------------------+
|                    详细结果表格                        |
|  序号 | 预测号码 | 开奖号码 | 中奖等级 | 命中详情    |
+-------------------------------------------------------+
```

### 5.5 Settings 设置页面

**布局结构**：
```
+-------------------------------------------------------+
|                    设置分类导航                        |
|  [系统设置] [预测配置] [显示设置] [关于系统]          |
+-------------------------------------------------------+
|                                                       |
|  系统设置:                                            |
|  - 数据源选择                                         |
|  - 缓存管理                                           |
|  - 硬件加速配置                                       |
|                                                       |
|  预测配置:                                            |
|  - 默认分析期数                                       |
|  - 默认预测方法                                       |
|  - 置信度阈值                                         |
|                                                       |
|  显示设置:                                            |
|  - 主题切换 (暗色/亮色)                               |
|  - 动画开关                                           |
|  - 语言选择                                           |
|                                                       |
+-------------------------------------------------------+
```

## 6. 动画规范

### 6.1 入场动画

```javascript
// 使用GSAP实现入场动画
import gsap from 'gsap'

// 页面入场
gsap.from('.page-content', {
  opacity: 0,
  y: 30,
  duration: 0.6,
  ease: 'power3.out'
})

// 卡片依次入场
gsap.from('.glass-card', {
  opacity: 0,
  y: 50,
  stagger: 0.1,
  duration: 0.8,
  ease: 'power3.out'
})

// 号码球弹出
gsap.from('.lottery-ball', {
  scale: 0,
  rotation: -180,
  stagger: 0.1,
  duration: 0.5,
  ease: 'back.out(1.7)'
})
```

### 6.2 交互动画

```css
/* 按钮悬停 */
.btn-primary {
  transition: all 0.3s ease;
}
.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
}

/* 卡片悬停 */
.glass-card {
  transition: all 0.3s ease;
}
.glass-card:hover {
  transform: translateY(-4px);
  border-color: rgba(0, 212, 255, 0.4);
}

/* 脉冲效果 */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
.pulsing {
  animation: pulse 2s infinite;
}
```

### 6.3 加载动画

```vue
<template>
  <div class="loading-overlay">
    <div class="loading-spinner">
      <div class="spinner-ring"></div>
      <div class="spinner-ring"></div>
      <div class="spinner-ring"></div>
    </div>
    <p class="loading-text">{{ message }}</p>
  </div>
</template>

<style scoped>
.spinner-ring {
  position: absolute;
  width: 60px;
  height: 60px;
  border: 3px solid transparent;
  border-radius: 50%;
  animation: spin 1.5s linear infinite;
}

.spinner-ring:nth-child(1) {
  border-top-color: #00d4ff;
  animation-delay: 0s;
}

.spinner-ring:nth-child(2) {
  border-right-color: #ff00ff;
  animation-delay: 0.2s;
}

.spinner-ring:nth-child(3) {
  border-bottom-color: #00ff88;
  animation-delay: 0.4s;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
```

## 7. API接口设计

### 7.1 预测相关

```typescript
// 获取预测方法列表
GET /api/predict/methods

// 执行预测
POST /api/predict
{
  method: string,
  periods: number,
  count: number,
  compound?: boolean,
  frontCount?: number,
  backCount?: number,
  acceleration?: string
}

// 获取预测进度 (SSE)
GET /api/predict/progress/:taskId
```

### 7.2 数据相关

```typescript
// 获取数据状态
GET /api/data/status

// 获取最新开奖
GET /api/data/latest

// 更新数据
POST /api/data/update

// 获取历史数据
GET /api/data/history?page=1&limit=20
```

### 7.3 分析相关

```typescript
// 频率分析
GET /api/analysis/frequency?periods=500

// 冷热分析
GET /api/analysis/hot-cold?periods=500

// 遗漏分析
GET /api/analysis/missing?periods=500
```

## 8. 目录结构

```
frontend/
|-- public/
|   |-- favicon.ico
|   |-- logo.svg
|-- src/
|   |-- api/                    # API接口
|   |   |-- index.ts
|   |   |-- predict.ts
|   |   |-- data.ts
|   |   |-- analysis.ts
|   |-- assets/                 # 静态资源
|   |   |-- images/
|   |   |-- fonts/
|   |-- components/             # 通用组件
|   |   |-- common/
|   |   |   |-- GlassCard.vue
|   |   |   |-- LotteryBall.vue
|   |   |   |-- AnimatedNumber.vue
|   |   |   |-- LoadingSpinner.vue
|   |   |-- layout/
|   |   |   |-- AppLayout.vue
|   |   |   |-- Sidebar.vue
|   |   |   |-- Header.vue
|   |   |-- charts/
|   |   |   |-- FrequencyChart.vue
|   |   |   |-- TrendChart.vue
|   |   |   |-- HeatmapChart.vue
|   |   |-- prediction/
|   |   |   |-- MethodSelector.vue
|   |   |   |-- ParamsPanel.vue
|   |   |   |-- ResultDisplay.vue
|   |   |   |-- ProgressAnimation.vue
|   |-- views/                  # 页面组件
|   |   |-- Dashboard.vue
|   |   |-- Prediction.vue
|   |   |-- Analysis.vue
|   |   |-- Compare.vue
|   |   |-- Settings.vue
|   |-- stores/                 # Pinia状态管理
|   |   |-- index.ts
|   |   |-- predict.ts
|   |   |-- data.ts
|   |   |-- settings.ts
|   |-- router/                 # 路由配置
|   |   |-- index.ts
|   |-- styles/                 # 全局样式
|   |   |-- variables.css
|   |   |-- global.css
|   |   |-- animations.css
|   |-- utils/                  # 工具函数
|   |   |-- request.ts
|   |   |-- format.ts
|   |   |-- animation.ts
|   |-- App.vue
|   |-- main.ts
|-- index.html
|-- vite.config.ts
|-- tailwind.config.ts
|-- tsconfig.json
|-- package.json
```

## 9. 性能优化

### 9.1 代码分割

```typescript
// 路由懒加载
const routes = [
  {
    path: '/',
    component: () => import('./views/Dashboard.vue')
  },
  {
    path: '/predict',
    component: () => import('./views/Prediction.vue')
  }
]
```

### 9.2 图片优化

- 使用WebP格式
- 实现懒加载
- 使用适当尺寸

### 9.3 缓存策略

- API响应缓存
- 静态资源长期缓存
- Service Worker支持

## 10. 响应式设计

### 10.1 断点设置

```css
/* Tailwind CSS断点 */
sm: 640px   /* 手机横屏 */
md: 768px   /* 平板 */
lg: 1024px  /* 小笔记本 */
xl: 1280px  /* 桌面 */
2xl: 1536px /* 大屏 */
```

### 10.2 移动端适配

- 侧边栏可折叠
- 表格横向滚动
- 触摸友好的按钮尺寸
- 简化的移动端布局

---

**文档版本**: v1.0.0
**创建日期**: 2024-12-09
**设计风格**: Cyberpunk + Glassmorphism
