import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="supermops",
    version="0.0.1",
    author="Alexander Schlüter",
    author_email="alx.schlueter@gmail.com",
    description="Super-resolution of moving point sources",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'cvxopt'],
    python_requires='>=3.5',
)