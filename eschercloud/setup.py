# setup.py

from setuptools import setup, find_packages

setup(
    name='eschercloud',
    version='0.1.4',
    packages=find_packages(),
    install_requires=[
        'requests',
    ],
    author="A.S",
    author_email="service@eschercloud.ai",
    description="Open source llm service"
    
)
