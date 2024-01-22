# from setuptools import find_packages,setup
# from typing import List

# HYPEN_E_DOT='-e .'
# def get_requirements(file_path:str)->List[str]:
#     '''
#     this function will return the list of requirements
#     '''
#     requirements=[]
#     with open(file_path) as file_obj:
#         requirements=file_obj.readlines()
#         requirements=[req.replace("\n","") for req in requirements]

#         if HYPEN_E_DOT in requirements:
#             requirements.remove(HYPEN_E_DOT)
    
#     return requirements

# setup(
# name='Transformer',
# version='0.0.1',
# author='Samir',
# author_email='samiraglarov0@gmail.com',
# packages=find_packages(),
# install_requires=get_requirements('requirements.txt')

# )
from setuptools import find_packages, setup
from setuptools.command.install import install
from typing import List
import subprocess

HYPEN_E_DOT = '-e .'


class CustomInstallCommand(install):
    def run(self):
        # Run the custom command before the actual installation
        subprocess.run(['TMPDIR=/home/user/tmp/ python3 -m pip install torch'], shell=True)
        # Continue with the default installation
        install.run(self)


def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='Transformer',
    version='0.0.1',
    author='Samir',
    author_email='samiraglarov0@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    cmdclass={'install': CustomInstallCommand}
)
