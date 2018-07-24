# MaD TwinNet Repository

### Welcome to the repository of the MaD TwinNet. 

If you know what you are doing, then jump ahead, get the **_pre-trained
weights_** from 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1164592.svg)](https://doi.org/10.5281/zenodo.1164592)
and start using the MaD TwinNet.  

If you just need the **_results_**, you can get them from
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1164585.svg)](https://doi.org/10.5281/zenodo.1164585)
.

There is also an on-line demo of the MaD TwinNet at the [website of
the MaD TwinNet](http://arg.cs.tut.fi/demo/mad-twinnet).

The paper of the MaD TwinNet is presented at the 
[IEEE World Congress on Computational Intelligence (WCCI)/International
Joint Conference on Neural Networks, 2018](http://www.ecomp.poli.br/~wcci2018/) and can be found
online at [the corresponding arXiv entry](https://arxiv.org/abs/1802.00300).

If you need some help on using MaD TwinNet, please read the following
instructions. 

Also, if you use any of the things existing in this repository
or the associated binary files from Zenodo, please consider 
citing [the MaD TwinNet paper, available from here](https://arxiv.org/abs/1802.00300).

## Previous work
A previous usage of the MaD architecture (including the **vanilla MaD**) can be found at:
https://github.com/Js-Mim/mss_pytorch

## Extensions
Joining forces in the research (apart from beers) with the colleague P. Magron, we enhanced
the performance of MaD TwinNet by applying phase recovery algorithms on top of MaD TwinNet.

We tried two use cases. One regards the singing voice separation. You can see the results 
for singing voice separation at the [corresponding online demo](http://arg.cs.tut.fi/demo/phase-madtwinnet/).
The results are also presented at [INTERSPEECH 2018](http://interspeech2018.org/) and the paper
can be found at [the paper entry on HAL (French equivalent of arXiv)](https://hal.archives-ouvertes.fr/hal-01741278).

The second case is for harmonic/percussive separation. The 
[corresponding online demo is here](http://arg.cs.tut.fi/demo/hpss-madtwinnet/), 
the paper is presented at the 
[16th International Workshop on Acoustic Signal Enhancement (IWAENC), 2018](http://www.iwaenc2018.org/), 
and an online version of the paper at arXiv is due to appear. You can see, though, the 
[HAL entry of the HPSS paper](https://hal.archives-ouvertes.fr/hal-01812225). 
  

## How do I use it with no manual
You can:

- re-train the MaD TwinNet, by running the script
[training.py](https://arg-redmine.tut.fi/projects/mad-twinnet/repository/revisions/master/entry/scripts/training.py),
- re-test it, by running the script 
[testing.py](https://arg-redmine.tut.fi/projects/mad-twinnet/repository/revisions/master/entry/scripts/testing.py), or 
- use it, by running the script 
[use_me.py](https://arg-redmine.tut.fi/projects/mad-twinnet/repository/revisions/master/entry/scripts/use_me.py).

The settings for all the above processes are controller by 
the [settings.py](https://arg-redmine.tut.fi/projects/mad-twinnet/repository/revisions/master/entry/helpers/settings.py) file. For the 
[use_me.py](https://arg-redmine.tut.fi/projects/mad-twinnet/repository/revisions/master/entry/scripts/use_me.py) script, you can find directions
in it or go to the How do I use it (RTFM version)? section.

If you want to re-test or use the MaD TwinNet, you will need 
the pre-trained weights of the MaD TwinNet. You can get the 
pre-trained weights from 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1164592.svg)](https://doi.org/10.5281/zenodo.1164592)
.  

## What is the MaD TwinNet
MaD TwinNet stands for the "Masker-Denoiser with Twin Networks
architecture/method" for monaural music sound source separation. 
An illustration of the MaD TwinNet can be seen in the following 
figure: 

![Illustration of MaD TwinNet](http://arg.cs.tut.fi/demo/mad-twinnet/assets/media/images/method.png)

You can read more at [our paper on arXiv](https://arxiv.org/abs/1802.00300). 

For the implementation of our method, we used the
[PyTorch framework](http://pytorch.org). 

## How do I use the MaD TwinNet
### Setting up the environment
Before starting using the code of this repository, you will have to
install some packages for your python environment. 

Must be noted that, our code is based on Python 3.6 version. So, for
a better experience, we recommend using Python 3.6. 

To install the dependencies, you can either use the `pip` package
manager or the `anaconda`/`conda`. 

If you want to **use the `pip`**, then you have to
- clone our repository (e.g. `git clone `),
- navigate with your terminal inside the directory of the cloned 
  repo (e.g. `cd mad-twinnet`), and then
- issue at your terminal the command 
`pip install -r requirements.txt` 

If you want to **use the `anaconda`/`conda`**, then you have to
- clone our repository (e.g. `git clone `),
- navigate with your terminal inside the directory of the cloned 
  repo (e.g. `cd mad-twinnet`), and then
- issue at your terminal the command 
`conda install --yes --file conda_requirements.txt`

### Dataset set-up
To do so, you will have to obtain your dataset. Your dataset should
be in the `dataset` directory. By default, the training set should
be under a directory named `Dev` and the testing set under a directory
named `Test`. This means that the directories for the training and
testing sets must be `dataset/Dev` and `dataset/Test`, respectively.

Also, by default, you will need numbered file names (e.g. `001.wav`)
and each file name should have an identifier whether the file is about
the mixture, the voice, the bass, and other. **Please check the 
[Demixing Secret Dataset (DSD)](http://www.sisec17.audiolabs-erlangen.de)
for the exact file naming conventions.** 

If you want to use the DSD, then you most probably will want to 
extract it in the `dataset` directory and you will end up with 
the above mentioned directory structure and proper file names.  

If you want to use a different dataset, then you have two options: 
- either you format your file names and directory structure to match
the one from the DSD, or
- you modify the file reading function to suit your needs.

For the second option, you will have to at least modify the 
`_get_files_lists` function, in the `helpers` directory/package.


### Using the pre-trained weights
To use the pre-trained weights of the MaD TwinNet, first you have
to obtain them from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1164592.svg)](https://doi.org/10.5281/zenodo.1164592)
.

Then, you have to unzip the obtained .zip file and move the resulting
files in the `outputs/states/` directory. These files will be the
following:
- rnn_enc.pt
- rnn_dec.pt
- fnn.pt
- denoiser.pt

You **must not** alter the names of the files and these files
cannot be used if you alter any members of the classes used
in the `modules/` directory. 

### Re-training MaD TwinNet
You can re-train the MaD TwinNet. For example, you might want to 
try and find better hyper-parameters, try how the MaD TwinNet will
go on a different training dataset, or any other wonderful idea :)

If you have set up the dataset correctly, then you just want to 
run the `scripts/training.py` file. You have quite enough options
to run this file. For example, you can run it through your favorite
IDE, or through terminal. 

If you run it through terminal, **please do not forget to set up
the PYTHONPATH environmental variable correctly**. E.g., if you are
in the project root directory, you can issue the command 
`export PYTHONPATH=$PYTHONPATH:../` and then you can issue the 
command `python scripts/training.py`. 

### Altering the hyper-parameters
All the hyper-parameters are in the `helpers/settings.py` file. 

You can alter any hyper-parameter you want, but make sure that 
the values that you will use are correct and can actually be used. 

### Re-testing MaD TwinNet
You can re-test the MaD TwinNet. To do so, you need again the proper
set-up of the dataset and the weights of the MaD TwinNet. 

When the above are OK, then you simply run the `scripts/testing.py`
file. 

If you run the testing file through terminal, **please do not forget to set up
the PYTHONPATH environmental variable correctly**. E.g., if you are
in the project root directory, you can issue the command 
`export PYTHONPATH=$PYTHONPATH:../` and then you can issue the 
command `python scripts/testing.py`.

### Use MaD TwinNet
To use the MaD TwinNet you need to have set up the pre-trained weights. 
If these weights are properly set up, then you need to call the script
`scripts/use_me.py` and provide as an argument:
- either a single file, or
- a text file (i.e. with ending .txt) which will have the path
to a single wav file in each line. 

The script will extract the voice and the background music from the provided
arguments (i.e. either the single wav file or all the wav files from the 
.txt file) and will save it as .wav file at the same position where the
corresponding wav file is. 

**Note bold:** All wav files must be 44.1 kHz sampling frequency and 16 bits
sample width (a.k.a. standard CD quality). 

Example of using the MaD TwinNet: 

`python scripts/use_me.py -w my_wav_file.wav`

or 

`python scripts/use_me.py -l a_txt_file_with_wavs.txt`

Please remember to set properly the python path 
(e.g. `export PYTHONPATH=$PYTHONPATH:../`)!

## Acknowledgements

- Part of the computations leading to these results was performed on a TITAN-X 
GPU donated by [NVIDIA](https://www.nvidia.com) to K. Drossos.
- K. Drossos and T. Virtanen wish to acknowledge [CSC-IT Center for Science, 
Finland](https://www.csc.fi/csc), for computational resources.
- D. Serdyuk  would  like  to  acknowledge the support of the following agencies for 
research funding and computing support: [Samsung](https://www.samsung.com), 
[NSERC](http://www.nserc-crsng.gc.ca/), 
[Calcul Quebec](http://www.calculquebec.ca/), 
[Compute Canada](https://www.computecanada.ca/),
the  [Canada  Research  Chairs](http://www.chairs-chaires.gc.ca/),  and  
[CIFAR](https://www.cifar.ca/).
- S.-I. Mimilakis is supported by the [European Union’s H2020  Framework  Programme
(H2020-MSCA-ITN-2014)](https://ec.europa.eu/programmes/horizon2020/) under grant 
agreement no [642685 MacSeNet](http://www.macsenet.eu/#0) (yes, in the video, is him).
- The authors would like to thank [P. Magron](http://www.cs.tut.fi/~magron/index.html) 
and 
[G. Naithani](https://tutcris.tut.fi/portal/en/persons/gaurav-naithani(93c5b9b8-53ec-4651-8b06-c8f764a3703e).html) (TUT, Finland) for their 
valuable comments and feedback during the writing process.