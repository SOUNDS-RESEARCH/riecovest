import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="riecovest",
    version="0.0.1",
    author="Jesper Brunnstrom",
    author_email="jesper.brunnstroem@kuleuven.be",
    description="Riemannian covariance estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
