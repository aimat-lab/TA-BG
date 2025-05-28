from codecs import open
from os import path

from setuptools import find_packages, setup

__version__ = "0.1"

here = path.abspath(path.dirname(__file__))

# Get the dependencies and installs:
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")
install_requires = [x.strip() for x in all_reqs]

setup(
    name="annealed_bg",
    version=__version__,
    description="Temperature-Annealed Boltzmann Generators",
    url="https://github.com/aimat-lab/TA-BG",
    license="MIT",
    keywords="",
    packages=find_packages(),
    include_package_data=True,
    author=["Henrik Schopmans, Pascal Friederich"],
    install_requires=install_requires,
    author_email="henrik.schopmans@kit.edu",
)
