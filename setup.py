from setuptools import find_packages, setup

setup(
    name='pm_icecon',
    version='0.2.0',
    description=('Sea ice concentration estimates from passive microwave data'),
    url='https://github.com/nsidc/pm_icecon',
    author='NSIDC Development Team',
    license='MIT',
    packages=find_packages(
        exclude=(
            '*.tasks',
            '*.tasks.*',
            'tasks.*',
            'tasks',
        ),
    ),
    include_package_data=True,
    zip_safe=False,
)
