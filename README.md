# Using Deep Learning for FACT Source Detection
This project was originally used for the thesis requirement for the Robert D. Clark Honors College, Department of Physics, and Department of Computer Science at the University of Oregon.

This project focuses on using convolutional neural networks to perform analysis of air shower events for the First G-APD Cherenkov Telescope (FACT), located in the Canary Islands.

# Organization

misc/ contains a (semi) organized collection of my attempts at using neural networks for air shower analysis. The misc/thesisFinal include the architectures that
were used for the actual thesis. The factnn/ is where all new development is happening, including making model creation more modular, easier generation of datasets,
and adding support for streaming in photon stream format files for both training and prediction. 

# Results

The quick results of this thesis were that convolutional neural networks did not beat the random forest method for 
separating gamma events from hadron events. After the conclusion of the thesis, I made networks that also took the time information, changing
the input from 2D image to a 3D cube. That improved the AUC of the separation up to .91, an improvement over the (as of this writing) 0.88 of the
current random forest method.

For estimating the energy of the initial particle, convolutional networks did almost as well, except for a much higher spread
at very high energies. 

Finally, for detecting the source position, neural networks did just as well or better than the current random forest method. The network including
time information achieved an R^2 = 0.77, while that one working with the 2D images had an R^2 = 0.60.

