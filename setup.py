from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tf-keras-vis",
    version="0.5.3",
    author="keisen",
    author_email="k.keisen@gmail.com",
    description="Neural network visualization toolkit for tf.keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keisen/tf-keras-vis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5, <3.9',
    install_requires=['numpy', 'scipy', 'imageio', 'pillow'],
    extras_require={
        'develop': ['flake8', 'isort', 'yapf', 'pytest<6.0', 'pytest-pep8', 'pytest-cov'],
        'examples': ['jupyterlab', 'matplotlib'],
    },
    include_package_data=True,
)
