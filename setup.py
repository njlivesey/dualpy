from setuptools import setup, find_packages

setup(
    name="dualpy",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "astropy",
        "pint",
        "numpy",
        "scipy",
        "sparse",
    ],
    python_requires=">=3.12",
)
