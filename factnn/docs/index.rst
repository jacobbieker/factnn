.. FACTNN documentation master file, created by
   sphinx-quickstart on Wed Oct 17 17:26:45 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FACTNN's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
========

FACTNN is a Python package designed to probe analyzing and classifying air shower data from the First G-APD Cherenkov Telescope (FACT)
using neural networks.

FACTNN can perform gamma/hadron separation, estimate the energy of gamma-ray initiated air showers, and estimate the source location of the air shower in the
night sky. This is thanks to training and testing data available from the FACT collaboration and hosted here: https://factdata.app.tu-dortmund.de/

If you want to just jump right in, examples are under the ```examples/``` folder, showing the process to build and run everything.

The design of this package is as follows.

 * The preprocessers are supposed to take a directory and convert it into the appropriate HDF5 file, storing it in an easy to access file.
 * The generators are supposed to, given a number of samples or files, infinitely generate batches for use in a neural network such as Tensorflow
 * The models are supposed to be a simpler way to create a model for this data based off of experiences with this as a thesis project. Subject to change.

Currently, the models are accting as they are supposed to, the generators work for the HDF5 inputs, but should do the processing for the files to stream them,
although that gives an overlap with the preprocessors then, the preprocessors are more supposed to generate a file and be done and not used later. The generators,
therefore, can take up the role of generating batches from files directly, including the preprocessing step. The generator should be given a list of indicies and HDF5 file, or
filepaths alone, and be able to generate batches for neural network.