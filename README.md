# Using Deep Learning for FACT Source Detection
This project was originally used for the thesis requirement for the Robert D. Clark Honors College, Department of Physics, and Department of Computer Science at the University of Oregon.

This project focuses on using convolutional neural networks to perform analysis of air shower events for the First G-APD Cherenkov Telescope (FACT), located in the Canary Islands.

# Organization

The final architectures that were used in the thesis are under the thesisFinal directory. 

Architectures that used the 2D image of each event are under flatModels/ and miscFailed/. Both those directories
include architectures that did not do well, or at all on the data.

Architectures that incorporate the time information are under timeModels/ and are generally the best performing architectures.

The preprocessing scripts for the FACT data are under preprocess/. This includes the rebinning and conversion to HDF5 files.

io/ contains the loading scripts for the timeModels and plotting scripts as well.

helpScripts/ includes the bash scripts that were used in this process to download data and run multiple jobs on Oregon's Talapas
supercomputer.

featureModels/ includes architectures that train on the parameterized images, the same as the random forest method.

testin/ includes the testing of some models.


# Results

The quick results of this thesis were that convolutional neural networks did not beat the random forest method for 
separating gamma events from hadron events. 

For estimating the energy of the initial particle, convolutional networks did almost as well, except for a much higher spread
at very high energies.

Finally, for detecting the source position, neural networks did just as well or better than the current random forest method.

