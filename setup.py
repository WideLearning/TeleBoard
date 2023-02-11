from setuptools import setup

setup(
    name="teleboard",
    version="0.0.1",
    install_requires=[
        "torch>=1.13",
        "matplotlib>=3.6.0",
        "PyQt5>=5.15.6",
        "neptune-client>=0.16.15",
    ]
)
