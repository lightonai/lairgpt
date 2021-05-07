import subprocess
from pathlib import Path

from setuptools import find_packages, setup

def version_from_git():
    """
    Get version from git tags
    git describe gives the most recent tag, and the number of commits
    between it and the current HEAD. Use it to output a version compatible
    with PEP440, <tag.postN>, with N being the number of commits from the tag.
    """
    if subprocess.run("git rev-parse --is-inside-work-tree".split(" "),
                      stdout=subprocess.PIPE).returncode != 0:
        return "0.0"

    git_describe = subprocess.run("git describe --tags --match [0-9]*".split(" "),
                                  check=True, stdout=subprocess.PIPE) \
        .stdout.decode("utf8").strip()
    t = git_describe.split("-")
    # If we're exactly at the git tag, there's no "-" and return the full string
    if len(t) == 1:
        return t[0]
    # If not, the number after is the number of commits to the latest tag
    if len(t) >= 2:
        return t[0]+".post"+t[1]
    raise RuntimeError("Failed to parse git describe: " + git_describe)


DESCRIPTION = "A Pytorch-based package by LightOn AI Research allowing to perform inference with PAGnol models."
__here__ = Path(__file__).absolute().parent

setup(
    name="lairgpt",
    version=version_from_git(),
    author="Lighton AI Research",
    author_email="iacopo@lighton.io,julien@lighton.io,igor@lighton.io",
    url="https://github.com/lightonai/lairgpt",
    description=DESCRIPTION,
    long_description=(__here__/"public"/"README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License"
    ],
    packages=find_packages(exclude=["build", "dist", "examples", "test*"]),
    install_requires=["pytorch==1.8.*", "omegaconf", "tokenizers", "python-wget"],
)
