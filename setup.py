from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="cnnClassifier",
    version="0.0.0",
    author="Kidney Disease Classification Team",
    author_email="het.sun04@gmail.com",
    description="Kidney Disease Classification using CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hetsevalia/Kidney-Disease-Classification-CICD-AWS-MLOps.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
)

