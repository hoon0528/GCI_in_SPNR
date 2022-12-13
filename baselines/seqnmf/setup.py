from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

readme = 'Python implementation of seqNMF.  For more information visit https://github.com/ContextLab/seqnmf.'

with open('LICENSE') as f:
    license = f.read()

setup(
    name='seqnmf',
    version='0.1.3',
    description='Python implementation of seqNMF',
    long_description=readme,
    author='Contextual Dynamics Laboratory',
    author_email='contextualdynamics@gmail.com',
    url='https://www.context-lab.com',
    license=license,
    install_requires=requirements,
    package_data = {'seqnmf':['data/MackeviciusData.mat']},
    packages=find_packages(exclude=('tests', 'docs'))
)
