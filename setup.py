from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="moment",
    version="0.1",
    description="MOMENT: A Family of Open Time-series Foundation Models",
    author="Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, Artur Dubrawski",
    author_email="mgoswami@andrew.cmu.edu",
    license="MIT",
    url="https://moment-timeseries-foundation-model.github.io/",
    zip_safe=False,
    packages=find_packages(exclude=["data", "tutorials"]),
    install_requires=required,
)
