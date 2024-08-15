from setuptools import find_packages, setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="momentfm",
    version="0.1.2",
    description="MOMENT: A Family of Open Time-series Foundation Models",
    author="Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, Artur Dubrawski",
    author_email="mgoswami@andrew.cmu.edu",
    license="MIT",
    url="https://moment-timeseries-foundation-model.github.io/",
    zip_safe=False,
    packages=find_packages(exclude=["data", "tutorials"]),
    install_requires=required,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
