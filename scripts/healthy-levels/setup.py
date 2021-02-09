# Package Based onahttps://raw.githubusercontent.com/probml/pyprobml/master/scripts/healthy_levels_plot.py
import setuptools
from setuptools import setup


setup(
    name="rectangle_estimation",
    version="0.0.2",
    author="Example Author",
    author_email="author@examples.com",
    description="A small example package",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.19.5",
        "matplotlib>=3.3.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
