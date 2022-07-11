import re
from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(req_path):
    """Read abstract requirements.
    Install requirements*.txt via pip first
    """
    # strip trailing whitespace, comments and URLs
    reqs = [
        re.sub(r"\s*([#@].*)?$", "", req)
        for req in Path(req_path).read_text().splitlines()
    ]
    # skip empty lines
    return [req for req in reqs if req]


setup(
    name="gi_tract_seg",
    packages=find_packages(),
    version="0.1.0",
    description="UW-Madison GI Tract Image Segmentation kaggle competition",
    author="skolchenko",
    license="MIT",
    # install_requires=read_requirements('requirements.txt'),
)
