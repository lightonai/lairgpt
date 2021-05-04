from setuptools import find_packages, setup

setup(
    name="lairgpt",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(exclude=["build", "dist", "examples", "test*"]),
)
