from setuptools import setup, find_packages

setup(
    name="food-segmentation-classifier",
    version="0.1.0",
    author="Food Vision Research",
    description="Advanced food portion size classifier using deep learning segmentation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "pillow>=8.3.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "tensorboard>=2.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "streamlit>=1.0.0",
        "ultralytics>=8.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    entry_points={
        "console_scripts": [
            "train-food-segmentation=training.train_swiss_7class:main",
            "monitor-training=utils.live_monitor:main",
        ]
    }
)