from setuptools import setup, find_packages

setup(
    name='vectordb_client',
    version='0.1.0',
    author="Raghvender",
    author_email="raghvender1205@gmail.com",
    description="Python SDK client for VectorDB",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Raghvender1205/VectorDBPythonClient",
    packages=find_packages(),
    install_requires=[
        "requests",
        "httpx",
        "pydantic"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)