#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大乐透智能预测系统 - 安装配置
"""

from setuptools import setup, find_packages
import os

# 读取README
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# 读取依赖
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='dlt-predictor',
    version='2.0.0',
    description='大乐透智能预测系统 - AI驱动的彩票预测平台',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='DLT Team',
    author_email='dlt@example.com',
    url='https://github.com/linshibo1994/dlt',
    license='MIT',

    packages=find_packages(exclude=['tests*', 'docs*']),
    include_package_data=True,

    python_requires='>=3.8',
    install_requires=read_requirements(),

    extras_require={
        'gui': ['streamlit>=1.0.0'],
        'gpu': ['tensorflow-gpu>=2.8.0'],
        'full': read_requirements(),
    },

    entry_points={
        'console_scripts': [
            'dlt=main:main',
            'dlt-cli=backend.app.main:main',
        ],
    },

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    keywords='lottery prediction machine-learning deep-learning',
)
