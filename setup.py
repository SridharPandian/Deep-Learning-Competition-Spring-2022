import os
import sys
from setuptools import find_packages, setup

print("Installing the Deep Learning SSL package.")

setup(
    name="dl-ssl",
    version='1.0.0',
    description="The package containing the training scripts for DL-SSL competition (Spring 2022).",
    author="abitha, sneha , sridhar",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "torch>=1.10.1+cu113",
        "torchvision==0.11.2+cu113",
        "byol-pytorch==0.5.7",
    ],
    python_requires=">=3.7",
)