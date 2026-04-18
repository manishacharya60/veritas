from setuptools import setup, find_packages

setup(
    name="veritas",
    version="0.1.0",
    description="VERITAS: Verified Exploration with Reflective Intelligent Tree-based Agents for Synthesis",
    author="",
    author_email="",
    url="https://github.com/[username]/VERITAS",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "full": [
            "vllm>=0.2.0",
            "lean-dojo>=1.0.0",
            "wandb>=0.15.0",
            "mlflow>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "veritas-evaluate=scripts.evaluate:main",
            "veritas-analyze=scripts.analyze:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
