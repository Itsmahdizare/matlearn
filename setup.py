import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="matlearn",
    version="1.0.0.0",
    author="Mahdi Zare",
    author_email="itsmahdizare@gmail.com",
    description="Matlearn is a machine learning library that enables create and train ML models and provides three major parts : preprocessing, training, and evaluation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/itsMahdiZare/matlearn",
    project_urls={
        "Bug Tracker": "https://github.com/itsMahdiZare/matlearn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["numpy"]
)