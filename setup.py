from setuptools import setup, find_packages


setup(
    name='enerflow',
    version='0.0.2',
    packages=find_packages(),
    license='MIT',
    description='⚡ Open-source framework for sequential decision problems in the energy domain',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='rebase.energy',
    author_email='hello@rebase.energy',
    url='https://github.com/rebase-energy/enerflow',
    install_requires=[
        "pandas",
        "pytz",
        "energydatamodel",
        "gymnasium",
    ]
)