from setuptools import setup, find_packages

setup(
    name="bridge-ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "rdflib>=6.0.0",
        "owlready2>=0.36"
    ],
    author="Michael Rushin",
    author_email="michael.r.rushin@gmail.com",
    description="Bi-directional Reasoning for Intelligence Data Graph Evolution Using Machine Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bridge-ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

