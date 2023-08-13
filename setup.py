from setuptools import setup, find_packages

setup(
    name='path_patching',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch'
    ],
    author='Callum McDougall',
    author_email='cal.s.mcdougall@gmail.com',  # Replace with your email
    description='A brief description of your project',  # Replace with your project's description
    url='https://github.com/callummcdougall/path_patching',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
