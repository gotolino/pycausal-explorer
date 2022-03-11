from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pycausal-explorer",
    version="0.1.0",
    url="https://github.com/gotolino/causal-learn",
    description="Python causal inference modules",
    packages=find_packages(exclude=["test*"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest >= 3.7",
            "check-manifest",
            "twine",
            "xgboost",
            "pre-commit >= 2.12",
            "pytest-cov >= 2.11",
            "flake8 >= 3.9",
            "mypy >= 0.910",
            "isort >= 5.9",
            "black >= 21.10b0",
        ],
    },
)
