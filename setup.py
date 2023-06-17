import setuptools

setuptools.setup(
    name='DIME',
    version='0.0.1',
    author='Soham Gadgil',
    author_email='sgadgil@cs.washington.edu',
    description='CMI Estimation',
    long_description='''
        Dynamic feature selection by estimation conditional mutual information.
    ''',
    long_description_content_type='text/markdown',
    url='',
    packages=['dime'],
    install_requires=[
        'numpy',
        'fastai>=2.5.3',
        'kaggle>=1.5.13',
        'opencv-python',
        'pandas',
        'scikit-learn',
        'tensorboard>=2.10.1',
        'timm>=0.6.12',
        'torch>=1.13.1',
        'torchvision>=0.14.1',
        'tqdm',
        'pandas>=1.5.2',
        'torchmetrics>=0.11.0',
        'pytorch_lightning==1.9.4'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.7',
)