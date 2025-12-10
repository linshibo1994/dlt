<template>
  <n-config-provider :theme="darkTheme" :theme-overrides="themeOverrides">
    <n-message-provider>
      <n-notification-provider>
        <n-dialog-provider>
          <AppLayout />
        </n-dialog-provider>
      </n-notification-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<script setup lang="ts">
import { darkTheme } from 'naive-ui'
import type { GlobalThemeOverrides } from 'naive-ui'
import AppLayout from './components/layout/AppLayout.vue'

// 自定义主题覆盖，匹配赛博朋克风格
const themeOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: '#00d4ff',
    primaryColorHover: '#33ddff',
    primaryColorPressed: '#00a8cc',
    primaryColorSuppl: '#00d4ff',
    successColor: '#00ff88',
    warningColor: '#ffaa00',
    errorColor: '#ff4757',
    textColorBase: '#ffffff',
    textColor1: 'rgba(255, 255, 255, 0.95)',
    textColor2: 'rgba(255, 255, 255, 0.82)',
    textColor3: 'rgba(255, 255, 255, 0.52)',
    textColorDisabled: 'rgba(255, 255, 255, 0.38)',
    placeholderColor: 'rgba(255, 255, 255, 0.38)',
    placeholderColorDisabled: 'rgba(255, 255, 255, 0.28)',
    iconColor: 'rgba(255, 255, 255, 0.52)',
    iconColorHover: 'rgba(255, 255, 255, 0.72)',
    iconColorPressed: 'rgba(255, 255, 255, 0.82)',
    iconColorDisabled: 'rgba(255, 255, 255, 0.28)',
    borderColor: 'rgba(255, 255, 255, 0.1)',
    dividerColor: 'rgba(255, 255, 255, 0.1)',
    inputColor: 'rgba(0, 0, 0, 0.3)',
    inputColorDisabled: 'rgba(0, 0, 0, 0.2)',
    bodyColor: '#0a0a1a',
    cardColor: 'rgba(255, 255, 255, 0.05)',
    modalColor: 'rgba(15, 15, 35, 0.95)',
    popoverColor: 'rgba(15, 15, 35, 0.95)',
    tableColor: 'rgba(0, 0, 0, 0.2)',
    actionColor: 'rgba(0, 0, 0, 0.2)',
    hoverColor: 'rgba(0, 212, 255, 0.1)',
    pressedColor: 'rgba(0, 212, 255, 0.15)',
    borderRadius: '8px',
    borderRadiusSmall: '6px',
    fontFamily: '"Inter", "PingFang SC", "Microsoft YaHei", sans-serif'
  },
  Button: {
    colorPrimary: '#00d4ff',
    colorHoverPrimary: '#33ddff',
    colorPressedPrimary: '#00a8cc',
    colorFocusPrimary: '#00d4ff',
    textColorPrimary: '#0a0a1a',
    textColorHoverPrimary: '#0a0a1a',
    textColorPressedPrimary: '#0a0a1a',
    textColorFocusPrimary: '#0a0a1a',
    borderPrimary: '1px solid #00d4ff',
    borderHoverPrimary: '1px solid #33ddff',
    borderPressedPrimary: '1px solid #00a8cc',
    borderFocusPrimary: '1px solid #00d4ff'
  },
  Card: {
    color: 'rgba(255, 255, 255, 0.05)',
    colorModal: 'rgba(15, 15, 35, 0.95)',
    borderColor: 'rgba(0, 212, 255, 0.2)',
    borderRadius: '12px'
  },
  Input: {
    color: 'rgba(0, 0, 0, 0.3)',
    colorFocus: 'rgba(0, 0, 0, 0.4)',
    borderColor: 'rgba(255, 255, 255, 0.1)',
    borderColorHover: 'rgba(0, 212, 255, 0.4)',
    borderColorFocus: '#00d4ff',
    placeholderColor: 'rgba(255, 255, 255, 0.38)'
  },
  Select: {
    peers: {
      InternalSelection: {
        color: 'rgba(0, 0, 0, 0.3)',
        colorActive: 'rgba(0, 0, 0, 0.4)'
      }
    }
  },
  Slider: {
    fillColor: '#00d4ff',
    fillColorHover: '#33ddff',
    handleColor: '#00d4ff'
  },
  Switch: {
    railColorActive: '#00d4ff'
  },
  Table: {
    thColor: 'rgba(0, 0, 0, 0.3)',
    tdColor: 'rgba(0, 0, 0, 0.2)',
    borderColor: 'rgba(255, 255, 255, 0.1)'
  },
  Tabs: {
    tabTextColorActiveLine: '#00d4ff',
    tabTextColorActiveBar: '#00d4ff',
    tabTextColorHoverLine: 'rgba(0, 212, 255, 0.8)',
    tabTextColorHoverBar: 'rgba(0, 212, 255, 0.8)',
    barColor: '#00d4ff'
  }
}
</script>

<style>
html, body, #app {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--bg-primary);
}

/* 全局滚动条样式 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.2);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: rgba(0, 212, 255, 0.3);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 212, 255, 0.5);
}

/* 全局选中文字颜色 */
::selection {
  background: rgba(0, 212, 255, 0.3);
  color: #ffffff;
}
</style>
