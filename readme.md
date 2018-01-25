# MaD TwinNet GitHub Repository

##Important!
Before you read anything else here, please make sure that you checked the licencing of this code. 

Also, if you use any of the things existing in this repository, please consider citing our paper: 

##Table of contents

- [How do I use it with no manual](mad-twinnet#How do I use it with no manual)
- [What is the MaD TwinNet](mad-twinnet#What is the MaD TwinNet)
- [How do I use the MaD TwinNet](mad-twinnet#How do I use the MaD TwinNet)
- [Acknowledgements](mad-twinnet#Acknowledgements)

##How do I use it with no manual
You can:

- re-train the MaD TwinNet, by running the script [training.py](scripts/training.py),
- re-test it, by running the script [testing.py](scripts/testing.py), or 
- use it, by running the script [use_me.py](scripts/use_me.py).

The settings for all the above processes are controller by the [settings.py](helpers/settings.py) file. 
For the [use_me.py](scripts/use_me.py) script, you can find directions in it or go to the 
How do I use it (RTFM version)? section.  

##MaD TwinNet, what is it?
MaD TwinNet stand for the Masker-Denoiser with Twin Networks architecture/method for monaural
music sound source separation. An illustration of the MaD TwinNet can be seen in the following figure: 

[Illustration of MaD TwinNet](images/method.png "Illustration of MaD TwinNet")

You can read more at our paper on arXiv: 

For the implementation of our method, we used the PyTorch framework. 

##How do I use the MaD TwinNet
###Re-training MaD TwinNet


