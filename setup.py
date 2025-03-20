from setuptools import setup, find_packages

setup(
    author="Peter Pak",
    name="amdt",
    version="0.0.3",
    # Loads in local packages
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "amdt": [
            "data/**/*",
        ]
    },
    entry_points={
        "console_scripts": [
            "amdt=amdt.manage:main",
        ]
    }
)
