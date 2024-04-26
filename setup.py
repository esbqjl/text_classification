from setuptools import setup, find_packages

# 读取 requirements.txt 中的依赖
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='emotion_classification',  
    version='0.0.1', 
    description='This project is a reasearch about machine learning vs. deep learning in emotion classification',  
    author='Wenjun Zhang',  
    author_email='1378555845gg@gmail.com',
    packages=find_packages(),  
    install_requires=required,  
    
)