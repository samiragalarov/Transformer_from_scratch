from setuptools import find_packages,setup
from typing import List
import subprocess



HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:


    index_url = "https://download.pytorch.org/whl/cpu"
    package = "torch"

    command = f"pip3 install {package} --index-url {index_url}"
    subprocess.run(command, shell=True)
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='Transformer',
version='0.0.1',
author='Samir',
author_email='samiraglarov0@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)
