from setuptools import setup, find_packages

setup(
    name="beamforge",
    version="1.0.0",
    packages=find_packages(include=["beamforge", "beamforge.*"]),
    python_requires=">=3.9",
    entry_points={"console_scripts": ["beamforge=beamforge.cli.main:main"]},
)
