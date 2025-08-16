from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="finance-capstone",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A finance-focused capstone project for portfolio analysis and risk management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/finance-capstone",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "yfinance>=0.1.70",
        "pandas-datareader>=0.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.12b0",
            "flake8>=4.0.1",
            "mypy>=0.910",
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
            "statsmodels>=0.13.0",
        ],
        "viz": [
            "plotly>=5.3.1",
            "cufflinks>=0.17.3",
        ],
        "web": [
            "streamlit>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="finance portfolio optimization risk management investment",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/finance-capstone/issues",
        "Source": "https://github.com/yourusername/finance-capstone",
    },
)
