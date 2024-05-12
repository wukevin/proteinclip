from setuptools import setup, find_packages

with open("README.md", "r") as source:
    long_description = source.read()

setup(
    name="proteinclip",
    author="Kevin Wu",
    packages=find_packages(),
    include_package_data=True,
    description="CLIP for protein sequences and descriptions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wukevin/proteinclip",
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "scipy",
        "scikit-learn",
        "pytorch-lightning",
        "seaborn",
        "mpl-scatter-density",
        "biotite",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    version="0.0.1",
)
