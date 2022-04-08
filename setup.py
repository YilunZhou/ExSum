from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'exsum',
    version = '1.0.0',
    author = 'Yilun Zhou',
    author_email = 'zyilun94@gmail.com',
    description = 'Explanation Summary (ExSum)',
    license = 'MIT',
    keywords = ['interpretability', 'natural language processing', 'machine learning'],
    url = 'https://yilunzhou.github.io/exsum/',
    packages=['exsum'],
    long_description = long_description,
    long_description_content_type='text/markdown', 
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Framework :: Flask', 
        'Intended Audience :: Science/Research', 
        'Topic :: Scientific/Engineering :: Artificial Intelligence', 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python', 
    ],
    entry_points = {
        'console_scripts': [
            'exsum=exsum.run_server:main'
        ]
    }, 
    install_requires = ['dill', 'Flask', 'numpy', 'scipy', 'tqdm'], 
    include_package_data = True
)
