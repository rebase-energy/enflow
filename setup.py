from setuptools import setup, find_packages


setup(
    name='enflow',
    version='0.0.2',
    packages=find_packages(),
    license='MIT',
    description='⚡ Open-source framework for sequential decision problems in the energy domain',
    author='rebase.energy',
    author_email='hello@rebase.energy',
    url='https://github.com/rebase-energy/enflow',
    install_requires=[
        "pandas",
        "pytz",
        "energydatamodel",
        "gymnasium",
    ]
)