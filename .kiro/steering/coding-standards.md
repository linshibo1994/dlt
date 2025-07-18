# Coding Standards

## Python Code Guidelines

### Code Style
- Follow PEP 8 conventions with 4-space indentation
- Use UTF-8 encoding with `# -*- coding: utf-8 -*-` header
- Include comprehensive docstrings for all classes and methods
- Use type hints for function parameters and return values
- Maximum line length of 120 characters

### Naming Conventions
- **Classes**: PascalCase (e.g., `BasicAnalyzer`, `CacheManager`)
- **Functions/Methods**: snake_case (e.g., `frequency_analysis`, `get_data`)
- **Variables**: snake_case (e.g., `front_balls`, `analysis_result`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_PERIODS`, `MAX_RETRIES`)
- **Private methods**: Leading underscore (e.g., `_load_analyzers`, `_ensure_cache_dirs`)

### Error Handling
- Use try-except blocks for external operations (file I/O, network requests)
- Log errors using the logger_manager with appropriate levels
- Return meaningful error messages and fallback values
- Handle pandas DataFrame operations with null checks

### Performance Patterns
- Use lazy loading for heavy modules (analyzers, predictors)
- Implement caching for expensive computations
- Use pandas vectorized operations instead of loops
- Employ progress bars for long-running operations

### Data Processing Standards
- **Lottery Data Format**: Front area (5 numbers, 1-35), Back area (2 numbers, 1-12)
- **Period Numbering**: Use string format for issue numbers (e.g., "24001", "24002")
- **Date Format**: Use pandas datetime format for consistency
- **Number Formatting**: Zero-pad display numbers (e.g., "07", "12")

### Module Structure Template
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module description
"""

import os
import sys
from typing import List, Dict, Tuple, Optional, Any

from core_modules import cache_manager, logger_manager, data_manager

class ModuleName:
    """Class description"""
    
    def __init__(self, param: str):
        self.param = param
    
    def method_name(self, arg: int) -> Dict[str, Any]:
        """Method description with parameters and return type"""
        try:
            # Implementation
            result = {}
            return result
        except Exception as e:
            logger_manager.error(f"Error in method_name: {e}")
            return {}
```

### Testing Patterns
- Include basic validation in methods (data existence, parameter ranges)
- Use assert statements for critical assumptions
- Implement graceful degradation when optional features fail
- Test with different data sizes (100, 500, 1000+ periods)