from setuptools import setup, find_packages

setup(
    name="aethelvision",
    version="1.0.0",
    description="Aethel Vision: Sovereign Clean-Room Edge AI Multi-Modal Platform",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Aethel AI Team",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=find_packages(include=["aethel*", "aethel_detector*"]),
    entry_points={
        "console_scripts": [
            "aethel=aethel.cli:main",
        ],
    },
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.23.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
)
