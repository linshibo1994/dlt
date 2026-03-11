#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data 子模块别名"""
import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_backend_app_dir = os.path.join(_project_root, 'backend', 'app')
if _backend_app_dir not in sys.path:
    sys.path.insert(0, _backend_app_dir)

_core_dir = os.path.join(_backend_app_dir, 'core')
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from backend.app.predictors.deep_learning.data import *
