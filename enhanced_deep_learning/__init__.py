#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
enhanced_deep_learning 兼容性别名包

此包是 backend.app.predictors.deep_learning 的别名，
用于保持向后兼容性，使得旧的导入语句仍然可以工作：
    from enhanced_deep_learning.models import LSTMPredictor
等价于：
    from backend.app.predictors.deep_learning.models import LSTMPredictor
"""

import sys
import os

# 将项目根目录添加到 Python 路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 添加 backend/app 目录到路径，使得 core_modules 等模块可以被找到
_backend_app_dir = os.path.join(_project_root, 'backend', 'app')
if _backend_app_dir not in sys.path:
    sys.path.insert(0, _backend_app_dir)

# 添加 backend/app/core 目录到路径
_core_dir = os.path.join(_backend_app_dir, 'core')
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

# 导入实际模块并重新导出
from backend.app.predictors.deep_learning import *
from backend.app.predictors.deep_learning import (
    __version__,
    __all__,
)

# 为了支持子模块导入，需要设置模块映射
# 这样 from enhanced_deep_learning.models import xxx 才能工作
