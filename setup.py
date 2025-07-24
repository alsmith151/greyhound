from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="greyhound",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning package for training models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/greyhound",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "tensorboard>=2.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.9",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "greyhound-train=greyhound.cli:train",
            "greyhound-eval=greyhound.cli:evaluate",
        ],
    },
)
