import subprocess
from os import path

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

    git_describe = subprocess.run("git describe --tags --match v[0-9]*".split(" "),
                                  check=True, stdout=subprocess.PIPE) \
        .stdout.decode("utf8").strip()
    t = git_describe.split("-")
    # If we're exactly at the git tag or after it
    # by some commits, only return a solid version
    if len(t) >= 1:
        return t[0]
    raise RuntimeError("Failed to parse git describe: " + git_describe)


DESCRIPTION = "A Pytorch-based package by LightOn AI Research allowing to perform inference with PAGnol models."
__here__ = path.abspath(path.dirname(__file__))
with open(path.join(__here__, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="lairgpt",
    version=version_from_git(),
    author="Lighton AI Research",
    author_email="iacopo@lighton.io,julien@lighton.io,igor@lighton.io",
    url="https://github.com/lightonai/lairgpt",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
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
    install_requires=[
        "torch==1.8.*",
        "tokenizers==0.10",
        "wget==3.2"
    ],
)
