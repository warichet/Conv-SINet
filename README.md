# Conv-SINet
Fully-convolutional speakers identification solution

General architecture:
* A specific encoder.
* A selection of points of interest in a time windows.
* A concatenation of results for a place.

Directories:
* Data is reponsible to get data-set and transform the data.
* Encoder is the deep learning part of the project that takes samples and produce vectors.
* Identification uses vectors to identify the speaker.
* Localization is in charge to determine the place of the speaker.

Data:
The data-set used is voxCeleb.
__getitem__() return (anchor, positive, negative) for the encoder part
__getitem__() return (label, sample) for the identification part

Encoder:
We compare two approaches, one that use STFT, the other is fully temporal. 
"TransFourier.py" is main STFT pyton file.
"Time.py" is main pyton file for time approach.
Freq directory is for notebook related to STFT approach. 
We try 2 ways for vectors distances (euclidian distance vs cosine).
We also try 3 differents sample size (1, 2 or 3 seconds).
Time directory is for notebook related to the fully temporal approach. 
We try 3 differents sample size (1, 2 or 3 seconds).

Identification:
"Conference.py" is a class that correspond to a conference.
"Place.py" is a class that correspond to a specific place in the conference room.
"Speaker.py" is a class that correspond to a speaker involved in the conference.
We continue to compare time vs STFT approach.
20 and 40 is for the nb of speaker in the conference.
"xxx_enc_1_xxx.ipynb" is for using encoder train with 1 second sample.
"xxx_sample_3_xxx.ipynb" is for using 3 seconds sample for the identification.
"xxx_speaker_xxx.ipynb" is for identify a speaker with one sample.
"Place_after_3_samples_xxx.ipynb" is for identify a speaker on a place with a concatenation of 3 samples.
"xxx_1_ref_in_pool.ipynb" is for identify speaker with one reference in the pool.

Localization:





