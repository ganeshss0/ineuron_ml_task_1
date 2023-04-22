from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:

    with open(file_path) as file:
        requirements = file.readlines()
    
    return [require.strip() for require in requirements if not HYPEN_E_DOT in require]


setup(
    name = 'Delivery_Time_Predict',
    version = '0.0.1',
    author = 'Ganesh',
    author_email = 'gs000@proton.me',
    install_requires = get_requirements('requirements.txt'),
    packages = find_packages()
)