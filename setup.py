import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Kidney-Disease-Classification-Deep-Learning-Project"
AUTHOR_USER_NAME = "Het Sevalia"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "het.sun04@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/hetsevalia/Kidney-Disease-Classification-CICD-AWS-MLOps.git",
    project_urls={
        "Bug Tracker": f"https://github.com/hetsevalia/Kidney-Disease-Classification-CICD-AWS-MLOps.git/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)