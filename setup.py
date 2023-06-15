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
    packages=['dime', 'experiments', 'baseline_models'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.7',
)