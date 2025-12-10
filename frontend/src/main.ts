import { createApp } from 'vue'
import { createPinia } from 'pinia'
import naive from 'naive-ui'
import router from './router'
import App from './App.vue'

// 导入样式
import './styles/variables.css'
import './styles/global.css'

const app = createApp(App)

// 注册插件
app.use(createPinia())
app.use(router)
app.use(naive)

app.mount('#app')
