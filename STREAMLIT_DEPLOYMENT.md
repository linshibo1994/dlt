# Streamlit Cloud 部署指南

## 🎯 部署问题分析

你遇到的错误 "installer returned a non-zero exit code" 通常是由于依赖包安装失败导致的。

## 📋 解决方案

### 1. 确认入口文件
**是的，你的项目入口文件确实是 `gui_app.py`**

所有启动脚本都指向这个文件：
- `start_gui.sh`: `python3 -m streamlit run gui_app.py`
- `run_gui.py`: `streamlit run gui_app.py`
- `Dockerfile`: `streamlit run gui_app.py`

### 2. 依赖问题解决

#### 方案A：使用精简版依赖（推荐）
使用新创建的 `requirements_streamlit.txt` 文件，它移除了重型依赖：

```bash
# 在Streamlit Cloud中指定这个文件作为requirements
requirements_streamlit.txt
```

#### 方案B：使用简化版应用
使用新创建的 `app.py` 作为入口文件，它是专门为Streamlit Cloud优化的版本。

### 3. Streamlit Cloud 部署步骤

1. **在GitHub上推送代码**
   ```bash
   git add .
   git commit -m "Add Streamlit Cloud deployment files"
   git push origin main
   ```

2. **在Streamlit Cloud中配置**
   - 仓库: 你的GitHub仓库
   - 分支: main
   - 主文件路径: `gui_app.py` 或 `app.py`（推荐使用app.py）
   - Python版本: 3.9

3. **高级设置**
   - Requirements file: `requirements_streamlit.txt`

### 4. 文件说明

#### 新创建的文件：
- `requirements_streamlit.txt`: 精简版依赖，移除了tensorflow、torch等重型包
- `.streamlit/config.toml`: Streamlit配置文件
- `app.py`: 简化版GUI应用，专门为云端部署优化
- `STREAMLIT_DEPLOYMENT.md`: 本部署指南

#### 原有文件：
- `gui_app.py`: 完整功能版本（适合本地运行）
- `requirements_gui.txt`: 已优化的GUI依赖文件

## 🚀 推荐部署方案

### 选项1：使用简化版（最稳定）
- 入口文件: `app.py`
- 依赖文件: `requirements_streamlit.txt`
- 优点: 部署成功率高，启动快
- 缺点: 功能相对简化

### 选项2：使用完整版（功能丰富）
- 入口文件: `gui_app.py`
- 依赖文件: `requirements_streamlit.txt`
- 优点: 功能完整
- 缺点: 可能因为模块依赖导致部署失败

## 🔧 故障排除

如果仍然部署失败：

1. **检查Python版本兼容性**
   - 确保使用Python 3.9或3.10
   - 避免使用Python 3.11+（某些包可能不兼容）

2. **进一步精简依赖**
   - 移除更多可选依赖
   - 只保留Streamlit核心功能所需的包

3. **检查代码兼容性**
   - 确保没有使用本地文件路径
   - 移除对本地模块的硬依赖

## 📝 部署检查清单

- [ ] 推送所有文件到GitHub
- [ ] 在Streamlit Cloud中选择正确的仓库和分支
- [ ] 设置主文件路径为 `app.py` 或 `gui_app.py`
- [ ] 设置requirements文件为 `requirements_streamlit.txt`
- [ ] 确认Python版本设置为3.9
- [ ] 检查部署日志中的具体错误信息

## 💡 建议

建议先使用 `app.py` + `requirements_streamlit.txt` 的组合进行部署，这是最稳定的方案。部署成功后，再考虑逐步添加更多功能。
