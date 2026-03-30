<template>
  <div class="history-page">
    <!-- 顶部统计与搜索 -->
    <GlassCard>
      <div class="toolbar">
        <div class="stats-info">
          <span class="stats-total">
            共 <strong>{{ total }}</strong> {{ isFiltered ? '条匹配结果' : '期历史数据' }}
          </span>
          <span v-if="total > 0" class="stats-range">
            当前显示第 {{ (currentPage - 1) * pageSize + 1 }} - {{ Math.min(currentPage * pageSize, total) }} 条
          </span>
        </div>
        <div class="search-area">
          <n-input
            v-model:value="searchIssue"
            placeholder="输入期号搜索..."
            clearable
            style="width: 200px"
            @keyup.enter="handleSearch"
          >
            <template #prefix>
              <SearchOutline class="input-icon" />
            </template>
          </n-input>
          <n-button type="primary" @click="handleSearch" :loading="isSearching">
            搜索
          </n-button>
          <n-button v-if="isFiltered" @click="handleReset">
            重置
          </n-button>
        </div>
      </div>
    </GlassCard>

    <!-- 数据列表 -->
    <GlassCard title="开奖记录" noPadding>
      <template #headerExtra>
        <n-button text type="primary" :loading="isLoading" @click="fetchData">
          <template #icon><RefreshOutline /></template>
          刷新
        </n-button>
      </template>

      <n-spin :show="isLoading">
        <div v-if="records.length > 0" class="record-list">
          <div
            v-for="record in records"
            :key="record.issue"
            class="record-item"
          >
            <div class="record-meta">
              <span class="record-issue">{{ record.issue }}</span>
              <span class="record-date">{{ record.date }}</span>
            </div>
            <div class="record-balls">
              <div class="ball-group">
                <span class="group-tag front-tag">前区</span>
                <div class="balls">
                  <LotteryBall
                    v-for="(num, idx) in record.front_balls"
                    :key="'f' + idx"
                    :number="num"
                    type="front"
                    size="sm"
                  />
                </div>
              </div>
              <div class="ball-group">
                <span class="group-tag back-tag">后区</span>
                <div class="balls">
                  <LotteryBall
                    v-for="(num, idx) in record.back_balls"
                    :key="'b' + idx"
                    :number="num"
                    type="back"
                    size="sm"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div v-else-if="!isLoading" class="empty-state">
          <n-empty description="暂无数据" />
        </div>
      </n-spin>
    </GlassCard>

    <!-- 分页 -->
    <div v-if="total > 0" class="pagination-wrapper">
      <n-pagination
        v-model:page="currentPage"
        v-model:page-size="pageSize"
        :item-count="total"
        :page-sizes="[20, 50, 100]"
        show-size-picker
        show-quick-jumper
        @update:page="handlePageChange"
        @update:page-size="handlePageSizeChange"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useMessage } from 'naive-ui'
import GlassCard from '@/components/common/GlassCard.vue'
import LotteryBall from '@/components/common/LotteryBall.vue'
import { SearchOutline, RefreshOutline } from '@vicons/ionicons5'
import { getHistoryData } from '@/api/data'
import type { LotteryResult } from '@/types'

const message = useMessage()

// 状态
const allFilteredRecords = ref<LotteryResult[]>([])
const totalFromServer = ref(0)
const currentPage = ref(1)
const pageSize = ref(20)
const isLoading = ref(false)
const isSearching = ref(false)
const searchIssue = ref('')
const isFiltered = ref(false)

// 当前页显示的记录（搜索模式下前端分页，正常模式下直接使用）
const serverRecords = ref<LotteryResult[]>([])

const records = computed(() => {
  if (isFiltered.value) {
    const start = (currentPage.value - 1) * pageSize.value
    return allFilteredRecords.value.slice(start, start + pageSize.value)
  }
  return serverRecords.value
})

const total = computed(() => {
  return isFiltered.value ? allFilteredRecords.value.length : totalFromServer.value
})

// 获取历史数据
const fetchData = async () => {
  try {
    isLoading.value = true
    const response = await getHistoryData(currentPage.value, pageSize.value)
    if (response.success) {
      serverRecords.value = response.data.records || []
      totalFromServer.value = response.data.total || 0
    } else {
      message.error(response.message || '获取数据失败')
    }
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : '未知错误'
    message.error('请求失败: ' + msg)
  } finally {
    isLoading.value = false
  }
}

// 搜索（在最近 200 期中前端过滤）
const handleSearch = async () => {
  if (isSearching.value) return
  const keyword = searchIssue.value.trim()
  if (!keyword) {
    if (isFiltered.value) {
      handleReset()
    }
    return
  }

  try {
    isSearching.value = true
    isLoading.value = true
    const response = await getHistoryData(1, 200)
    if (response.success) {
      const allRecords = response.data.records || []
      const filtered = allRecords.filter((r: LotteryResult) =>
        r.issue.includes(keyword)
      )
      allFilteredRecords.value = filtered
      currentPage.value = 1
      isFiltered.value = true
      if (filtered.length === 0) {
        message.info('未在最近 200 期中找到匹配结果')
      }
    } else {
      message.error(response.message || '搜索失败')
    }
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : '未知错误'
    message.error('搜索失败: ' + msg)
  } finally {
    isSearching.value = false
    isLoading.value = false
  }
}

// 重置搜索
const handleReset = () => {
  searchIssue.value = ''
  isFiltered.value = false
  allFilteredRecords.value = []
  currentPage.value = 1
  fetchData()
}

// 翻页
const handlePageChange = (page: number) => {
  currentPage.value = page
  if (!isFiltered.value) {
    fetchData()
  }
}

// 每页条数变化
const handlePageSizeChange = (size: number) => {
  pageSize.value = size
  currentPage.value = 1
  if (!isFiltered.value) {
    fetchData()
  }
}

onMounted(() => {
  fetchData()
})
</script>

<style scoped>
.history-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 工具栏 */
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}

.stats-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stats-total {
  font-size: 15px;
  color: var(--text-primary);
}

.stats-total strong {
  color: var(--neon-blue);
  font-size: 18px;
}

.stats-range {
  font-size: 12px;
  color: var(--text-tertiary);
}

.search-area {
  display: flex;
  align-items: center;
  gap: 8px;
}

.input-icon {
  width: 16px;
  height: 16px;
  color: var(--text-tertiary);
}

/* 记录列表 */
.record-list {
  display: flex;
  flex-direction: column;
}

.record-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border-default);
  transition: background var(--transition-fast);
}

.record-item:last-child {
  border-bottom: none;
}

.record-item:hover {
  background: rgba(0, 212, 255, 0.05);
}

.record-meta {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 100px;
}

.record-issue {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.record-date {
  font-size: 12px;
  color: var(--text-tertiary);
}

/* 号码球 */
.record-balls {
  display: flex;
  align-items: center;
  gap: 24px;
}

.ball-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.group-tag {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  white-space: nowrap;
}

.front-tag {
  background: rgba(0, 212, 255, 0.15);
  color: var(--neon-blue);
  border: 1px solid rgba(0, 212, 255, 0.3);
}

.back-tag {
  background: rgba(255, 0, 255, 0.15);
  color: var(--neon-pink);
  border: 1px solid rgba(255, 0, 255, 0.3);
}

.balls {
  display: flex;
  gap: 4px;
}

/* 空状态 */
.empty-state {
  padding: 60px 0;
}

/* 分页 */
.pagination-wrapper {
  display: flex;
  justify-content: center;
  padding: 8px 0;
}

/* 响应式 */
@media (max-width: 768px) {
  .toolbar {
    flex-direction: column;
    align-items: stretch;
  }

  .search-area {
    flex-wrap: wrap;
  }

  .record-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
  }

  .record-balls {
    flex-wrap: wrap;
    gap: 12px;
  }
}
</style>
