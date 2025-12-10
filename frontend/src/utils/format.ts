// 格式化数字
export const formatNumber = (num: number, digits = 2): string => {
  return num.toString().padStart(digits, '0')
}

// 格式化日期
export const formatDate = (date: string | Date, format = 'YYYY-MM-DD'): string => {
  const d = new Date(date)
  const year = d.getFullYear()
  const month = formatNumber(d.getMonth() + 1)
  const day = formatNumber(d.getDate())

  return format
    .replace('YYYY', year.toString())
    .replace('MM', month)
    .replace('DD', day)
}

// 格式化货币
export const formatCurrency = (amount: number): string => {
  return new Intl.NumberFormat('zh-CN', {
    style: 'currency',
    currency: 'CNY'
  }).format(amount)
}

// 格式化百分比
export const formatPercent = (value: number, digits = 1): string => {
  return `${(value * 100).toFixed(digits)}%`
}

// 格式化大数字
export const formatLargeNumber = (num: number): string => {
  if (num >= 100000000) {
    return `${(num / 100000000).toFixed(2)}亿`
  }
  if (num >= 10000) {
    return `${(num / 10000).toFixed(2)}万`
  }
  return num.toLocaleString()
}
