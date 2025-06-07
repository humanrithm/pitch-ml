from setuptools import setup, find_packages

setup(
    name="pitch_ml",
    version="0.0.5",
    packages=find_packages(where="packages"),
    package_dir={"": "packages"},
    python_requires=">=3.8",
    install_requires=[
        'pandas>=2.0',
        'torch>=2.2.2',
        'dropbox>=12.0.2',
        'scipy>=1.14.0',
        'matplotlib>=3.4.0',
        'psycopg2 >= 2.9.9',
        'python-dotenv>=1.1.0',
        'sshtunnel>=0.4.0',
        'boto3>=1.38.25'
    ],
    author='Connor Moore',
    author_email='rcm8@clemson.edu',
    description='Pitch ML: Packages for optimizing pitcher health and performance.',
    url='https://github.com/humanrithm/pitch-ml',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
    ],
)