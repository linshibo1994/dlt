# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

大乐透智能预测系统 - A Chinese lottery prediction system with 26+ algorithms including deep learning (TensorFlow), traditional statistics, probability models, and adaptive learning based on 2756+ periods of historical data.

## Development Commands

### Running the Application

```bash
# GUI (primary interface) - opens at http://localhost:8501
python frontend/run.py
# or
python main.py --gui

# Command Line Interface
python main.py predict -m <method> -p <periods> -c <count>
# Example: python main.py predict -m markov -p 500 -c 3
```

### Testing

```bash
pytest                           # Run all tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests only
pytest -v --tb=short            # Verbose with short traceback
python -m py_compile backend/app/main.py  # Syntax check
```

### Data Management

```bash
python main.py data status       # Check data status
python main.py data update       # Update lottery data
python main.py data latest       # Get latest draw result
```

## Architecture

### Directory Structure

```
backend/
├── app/
│   ├── main.py              # CLI entry point & DLTPredictorSystem class
│   ├── core/
│   │   ├── core_modules.py  # cache_manager, logger_manager, data_manager, task_manager
│   │   ├── path_config.py   # Centralized path management (PathConfigManager)
│   │   └── smart_cache_system.py
│   ├── predictors/
│   │   ├── predictor_modules.py  # TraditionalPredictor, AdvancedPredictor, SuperPredictor
│   │   ├── compound/             # Compound betting (复式投注)
│   │   ├── deep_learning/        # LSTM, Transformer, GAN models
│   │   ├── markov/               # Markov chain predictors
│   │   └── traditional/          # Statistical predictors
│   ├── analyzers/                # BasicAnalyzer, AdvancedAnalyzer
│   ├── learning/                 # Adaptive learning modules
│   └── improvements/             # Enhanced Markov, ensemble methods
├── api/                          # REST API (if needed)
frontend/
├── streamlit/                    # Streamlit GUI components
└── run.py                        # GUI launcher
config/
├── prediction.yaml              # Algorithm settings
├── training.yaml                # ML training parameters
├── acceleration.yaml            # GPU/CPU acceleration settings
└── paths.yaml                   # Path configurations
data/
└── dlt_data_all.csv             # Historical lottery data (2756+ periods)
tests/
├── unit/                        # Unit tests
├── integration/                 # Integration tests
└── predictor/                   # Predictor-specific tests
```

### Key Classes

**DLTPredictorSystem** (`backend/app/main.py`):
- Main orchestrator class with lazy-loaded predictors and analyzers
- Methods: `run_predict_command()`, `run_analyze_command()`, `run_data_command()`
- Uses `OutputStatus` class for unified status output constants

**PathConfigManager** (`backend/app/core/path_config.py`):
- Singleton for centralized path management
- All paths should use `get_path_manager()` instead of hardcoded paths

**Predictors** (`backend/app/predictors/predictor_modules.py`):
- `TraditionalPredictor`: frequency, hot_cold, missing, bayesian
- `AdvancedPredictor`: markov, ensemble, clustering, nine_models
- `SuperPredictor`: super, adaptive predictions

### Algorithm Categories

1. **Traditional Statistics**: frequency, hot_cold, missing, bayesian
2. **Markov Chains**: markov, markov_2nd, markov_3rd, adaptive_markov
3. **Deep Learning**: lstm, transformer, gan, ensemble
4. **Intelligent**: super, adaptive, nine_models, highly_integrated
5. **Compound Betting**: compound, duplex, markov_compound

## Code Patterns

### Module Loading
System uses lazy loading - core modules load first, enhanced features (deep learning) load on demand:
```python
# Correct pattern
self._load_predictors()  # Call before using self.predictors
```

### Configuration Access
```python
from core.path_config import get_path_manager
pm = get_path_manager()
data_file = pm.data_file  # Use PathConfigManager for all paths
```

### Status Output
```python
from main import OutputStatus
print(f"{OutputStatus.OK} Operation completed")
print(f"{OutputStatus.ERROR} Something failed")
```

### Error Handling
System has graceful degradation - GPU failures fall back to CPU, deep learning failures fall back to traditional methods.

## Dependencies

- Python 3.8+ (3.10+ recommended)
- TensorFlow 2.8+ (optional, for deep learning)
- Streamlit (GUI)
- pandas, numpy, scikit-learn (data processing)

See `requirements.txt` for full list.
