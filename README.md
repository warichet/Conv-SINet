# Conv-SINet
Fully-convolutional speakers identification solution

General architecture:
* A specific encoder.
* A selection of point of interest in a time windows.
* A concatenation of results for a place.

Directories:
* Data is reponsible to get data-set and transform the data.
* Encoder is the deep learning part of the project.
* Identification is treatement of vector produced by theencoder.
* Localization is in charge to determine the place of the speaker.
