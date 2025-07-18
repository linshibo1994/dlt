# Architecture Guidelines

## System Architecture Principles

### Modular Design
- **Core Modules** (`core_modules.py`): Data management, caching, logging, progress tracking
- **Analyzer Modules** (`analyzer_modules.py`): Basic, advanced, and comprehensive analysis
- **Predictor Modules** (`predictor_modules.py`): Traditional, advanced, and super predictors
- **Adaptive Learning** (`adaptive_learning_modules.py`): Multi-armed bandit algorithms
- **Main Entry** (`dlt_main.py`): Unified CLI interface with lazy loading

### Performance Optimization
- **Lazy Loading**: Modules loaded on-demand to reduce startup time
- **Intelligent Caching**: Analysis results and models cached for performance
- **Progress Management**: Real-time progress display with interrupt/resume support
- **Memory Efficiency**: Efficient data structures and garbage collection

### Data Flow Architecture
```
Data Sources (中彩网 API) → Data Manager → Cache Manager → Analyzers → Predictors → Output
                                    ↓
                              Adaptive Learning ← Performance Tracking
```

### Key Design Patterns
- **Strategy Pattern**: Multiple prediction algorithms with unified interface
- **Observer Pattern**: Progress tracking and logging
- **Factory Pattern**: Predictor and analyzer instantiation
- **Singleton Pattern**: Cache and data managers
- **Command Pattern**: CLI command handling

### Technology Stack
- **Core**: Python 3.8+, pandas, numpy, scipy
- **ML/DL**: scikit-learn, tensorflow, keras, xgboost, lightgbm
- **Analysis**: matplotlib, seaborn, networkx
- **Data**: requests, beautifulsoup4, lxml
- **Bayesian**: pymc, arviz (optional)

### File Organization
- Configuration files in root directory
- Data files in `data/` directory
- Cache files in `cache/` with subdirectories by type
- Output files in `output/` with timestamp-based naming
- Logs in `logs/` directory with rotation