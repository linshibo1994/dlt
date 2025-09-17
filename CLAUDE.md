# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese lottery prediction system (大乐透智能预测系统) built with Python. It's an AI-powered prediction platform that uses 26+ algorithms including deep learning (TensorFlow), traditional statistics, probability models, and adaptive learning to predict lottery numbers based on historical data from 2756 periods.

## Development Commands

### Running the Application

**GUI Application (Primary Interface):**
```bash
# Linux/Mac
./start_gui.sh [--host HOST] [--port PORT] [--localhost-only]

# Windows
start_gui.bat

# Manual start
python -m streamlit run gui_app.py --server.port 8501 --server.address localhost
```

**Command Line Interface:**
```bash
python dlt_main.py [options]
```

### System Management

**Environment Check:**
```bash
python system_check.py
```

**Deploy to Production:**
```bash
./deploy.sh          # Full production deployment with nginx
./quick_deploy.sh     # Docker-based deployment
./one_click_deploy.sh # Quick local deployment
```

**View Logs:**
```bash
./logs_viewer.sh
```

### Testing and Quality

The project uses pytest for testing. Tests can be found by searching for files containing "test" or "Test".

```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest --cov             # With coverage
```

Code quality tools are configured in requirements.txt:
```bash
black .                   # Code formatting
isort .                   # Import sorting
flake8 .                  # Linting
mypy .                    # Type checking
```

## Architecture

### Core System Structure

**Main Entry Points:**
- `dlt_main.py` - Command-line interface and core system orchestration
- `gui_app.py` - Streamlit-based web GUI interface
- `run_gui.py` - GUI launcher wrapper

**Core Modules:**
- `core_modules.py` - System foundation (cache, logging, data management, task management)
- `predictor_modules.py` - All 26+ prediction algorithms implementation
- `analyzer_modules.py` - Data analysis and statistical processing
- `adaptive_learning_modules.py` - Self-learning and optimization systems

**Enhanced Features:**
- `enhanced_integration.py` - Advanced feature integration
- `enhanced_deep_learning/` - Deep learning models (LSTM, Transformer, GAN)
- `smart_cache_system.py` - Intelligent caching with 90.6x performance improvement

### Data Architecture

**Data Storage:**
- `data/dlt_data_all.csv` - Complete historical lottery data (2756+ periods)
- `cache/` - Intelligent caching system with data version control
- `models/` - Trained ML models (LSTM, Transformer, ensemble models)
- `config/` - System configuration files

**Configuration Files:**
- `config/config.json` - Main system configuration
- `config/prediction.yaml` - Prediction algorithm settings
- `config/training.yaml` - ML model training parameters
- `config/acceleration.yaml` - Hardware acceleration settings
- `config/gui_config.json` - GUI interface settings

### Algorithm Categories

1. **Traditional Statistics** (频率分析, 冷热号分析, 遗漏值分析, 贝叶斯分析)
2. **Markov Chains** (1-3阶马尔可夫链, 自适应马尔可夫)
3. **Deep Learning** (LSTM时序预测, Transformer注意力, GAN生成对抗, 集成深度学习)
4. **Intelligent Prediction** (自适应预测, 超级预测, 9种数学模型)
5. **Compound Betting** (标准复式, 胆拖投注, 高级复式)

### System Features

**Performance Optimization:**
- Intelligent GPU/CPU acceleration with hardware detection
- Smart caching system with 90.6x performance improvement
- Memory optimization (peak 249.8MB)
- All algorithms execute in <1 second

**Data Processing:**
- Support for 50-2756 analysis periods
- 1-100 number generation options
- 7 betting methods (单式, 复式, 胆拖, etc.)
- Automatic data crawling and updates

## Dependencies

**Core Requirements:**
- Python 3.8+ (recommended 3.10+)
- TensorFlow 2.8+ (deep learning)
- Streamlit (GUI framework)
- See `requirements.txt` for complete dependency list

**Optional GPU Support:**
- CUDA-enabled GPU for TensorFlow acceleration
- Apple Silicon (Metal) support for M1/M2 Macs

## Development Tips

1. **Module Loading:** The system uses lazy loading to avoid startup delays. Core modules are loaded first, enhanced features are optional.

2. **Error Handling:** The system has comprehensive error handling and graceful degradation (e.g., GPU→CPU fallback).

3. **Caching:** Heavy use of intelligent caching - be aware of cache invalidation when modifying data processing logic.

4. **Algorithms:** All 26+ algorithms are complete mathematical implementations, not simplified versions. They're in `predictor_modules.py`.

5. **Configuration:** System behavior is heavily configurable through YAML/JSON files in `config/` directory.

6. **Hardware Acceleration:** The system automatically detects and optimizes for available hardware (CPU cores, GPU, memory).

## File Structure Notes

- Shell scripts (`.sh`) are for Linux/Mac deployment and management
- Batch files (`.bat`) are for Windows users
- `enhanced_deep_learning/` contains advanced ML implementations
- `compound_modules/` contains compound betting logic
- `improvements/` likely contains system enhancements and optimizations