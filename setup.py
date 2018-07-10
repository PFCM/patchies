"""packaging"""
from setuptools import setup


def read_file(path):
    """Read a while file into a string"""
    with open(path) as infile:
        return infile.read()


setup(
    name='patchies',
    version='0.1',
    py_modules=['patchies'],
    install_requires=read_file('requirements.txt').split('\n'),
    entry_points='''
        [console_scripts]
        patchies=patchies.pipeline:run
        ''')
