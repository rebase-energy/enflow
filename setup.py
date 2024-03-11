from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as file:
        return file.read().splitlines()

setup(
    name='enerflow',
    version='0.0.1',
    packages=find_packages(),
    license='MIT',
    description='⚡ Open-source framework for sequential decision problems in the energy domain',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='rebase.energy',
    author_email='hello@rebase.energy',
    url='https://github.com/rebase-energy/enerflow',
    install_requires=read_requirements()
)