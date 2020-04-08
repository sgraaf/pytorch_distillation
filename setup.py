import setuptools

__version__ = '0.0.1'
url = 'https://github.com/sgraaf/pytorch_distillation'

with open('README.md', 'r', encoding='utf-8') as f:
    README = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='torch_distillation',
    version=__version__,
    license='GNU General Public License (GNU GPL v3 or above)',
    author='Steven van de Graaf',
    author_email='steven@vandegraaf.xyz',
    description='Knowledge Distillation Extension Library for PyTorch',
    long_description=README,
    long_description_content_type='text/markdown',
    keywords='pytorch knowledge-distillation distillation',
    url=url,
    download_url=f'{url}/{__version__}.tar.gz',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
    ],
)