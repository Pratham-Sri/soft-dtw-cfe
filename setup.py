from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="soft_dtw_cfe",
    version="1.0.0",
    author="Pratham Srivastava / based on Kostrzewa, Galus, Zięba (2026)",
    description="Soft-DTW Counterfactual Explanations for Time Series Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pratham-Sri/soft-dtw-cfe",
    packages=[
        "soft_dtw_cfe",
        "soft_dtw_cfe.data",
        "soft_dtw_cfe.evaluation",
        "soft_dtw_cfe.methods",
        "soft_dtw_cfe.methods.dtw_guided",
        "soft_dtw_cfe.models",
        "soft_dtw_cfe.visualization",
    ],
    package_dir={
        "soft_dtw_cfe": ".",
        "soft_dtw_cfe.data": "data",
        "soft_dtw_cfe.evaluation": "evaluation",
        "soft_dtw_cfe.methods": "methods",
        "soft_dtw_cfe.methods.dtw_guided": "methods/dtw_guided",
        "soft_dtw_cfe.models": "models",
        "soft_dtw_cfe.visualization": "visualization",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
)
