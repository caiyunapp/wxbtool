import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wxbtool",
    version="0.0.51",
    author="Mingli Yuan",
    author_email="mingli.yuan@gmail.com",
    description="A toolkit for WeatherBench based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/caiyunapp/wxbtool",
    project_urls={
        'Documentation': 'https://packaging.python.org/tutorials/distributing-packages/',
        'Source': 'https://github.com/caiyunapp/wxbtool',
        'Tracker': 'https://github.com/caiyunapp/wxbtool/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'torch',
        'numpy',
        'netcdf4',
        'toolz',
        'dask',
        'xarray',
        'opencv-python',
        'arrow',
        'python-decouple',
        'leibniz',
        'flask',
        'msgpack',
        'msgpack-numpy',
        'requests',
        'gunicorn',
        'arghandler',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points={
        'console_scripts': [
            'wxb = wxbtool.wxb:main',
        ]
    }
)

