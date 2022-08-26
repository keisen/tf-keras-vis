from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="tf-keras-vis",
    version=version,
    author="keisen (Yasuhiro Kubota)",
    author_email="k.keisen@gmail.com",
    description="Neural network visualization toolkit for tf.keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/keisen/tf-keras-vis",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.7',
    install_requires=[
        'scipy',
        'pillow',
        'deprecated',
        'imageio',
        'packaging',
        'importlib-metadata; python_version < "3.8"',
    ],
    extras_require={
        'develop': [
            'flake8',
            'flake8-docstrings',
            'isort',
            'yapf',
            'pytest',
            'pytest-pycodestyle',
            'pytest-cov',
            'pytest-env',
            'pytest-xdist',
        ],
        'docs': [
            'sphinx',
            'sphinx-autobuild',
            'sphinx-rtd-theme',
            'myst-parser',
            'nbsphinx',
            'pandoc',
        ],
        'examples': [
            'jupyterlab',
            'matplotlib',
        ],
    },
    include_package_data=True,
)
