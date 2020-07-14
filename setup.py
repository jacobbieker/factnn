from setuptools import setup, find_packages

setup(
    name="factnn",
    version="0.5",
    license="GPLv3",
    author="Jacob Bieker",
    authoer_email="jacob.bieker@gmail.com",
    url="https://github.com/jacobbieker/factnn",
    download_url="https://github.com/jacobbieker/factnn/archive/v0.5.0.tar.gz",
    keywords=["IACT Astronomy", "FACT", "Machine Learning", "Tensorflow"],
    packages=find_packages(),
    install_requires=["astropy", "numpy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
    ],
)
