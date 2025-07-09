from setuptools import setup, find_packages

setup(
    name="abap_transform",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "pytest>=7.0.0",
    ],
) 