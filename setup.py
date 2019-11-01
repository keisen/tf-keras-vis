import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tf-keras-vis",
    version="0.0.1",
    author="Keisen",
    author_email="k.keisen@gmail.com",
    description="Neural network visualization toolkit for tf.keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keisen/tf-keras-vis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
