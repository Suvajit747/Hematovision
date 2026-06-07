from setuptools import setup, find_packages

setup(
    name="hematovision",
    version="1.0.0",
    description="Advanced blood cell classification using transfer learning",
    author="HematoVision Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.12",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.1",
        "gradio>=4.0.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "isort", "flake8"],
        "viz": ["tensorboard>=2.14.0", "grad-cam>=1.4.8"],
    },
    entry_points={
        "console_scripts": [
            "hematovision-train=train:main",
            "hematovision-eval=evaluate:main",
            "hematovision-predict=predict:main",
            "hematovision-demo=demo.app:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
