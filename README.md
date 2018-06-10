# Automatic Speech Recognition for Speech to Text on Chinese

This is my [Google Summer of Code 2018 Project](https://summerofcode.withgoogle.com/projects/#5284664500027392) with the [Red Hen Lab](http://www.redhenlab.org/).

The aim of this project is to develop a working Speech-to-Text module for the Red Hen Lab’s Chinese Pipeline, resulting in a working application. The initial goal is to establish a Tensorflow implementation for Chinese speech recognition based on [Mozilla's DeepSpeech](https://github.com/mozilla/DeepSpeech). During the GSoC coding period, we've found a better option for Chinese ASR: an open source program named [DeepSpeech2 on PaddlePaddle](https://github.com/PaddlePaddle/DeepSpeech) based on [DeepSpeech2 Paper](http://proceedings.mlr.press/v48/amodei16.pdf), which better suit Chinese Pipeline rather than Mozilla’s DeepSpeech. Until the end of the 4th week in GSoC, I have progressed to the point of being able to run DeepSpeech2 on PaddlePaddle inside Singularity on CWRU HPC and already had a perfect model developed by Baidu with its abundant Chinese materials.

Based on these, I make a slight change for my future tasks:
1. Fully understand the code and model framework provided by Baidu.
2. Use the model in DeepSpeech2 to test its word error rate and study on its usability.
3. Script using Shell and Python to make the workflow better fit in Red Hen's pipeline.

#### Contents

1. [Getting Started](#getting-started)
2. [Data-Preprocessing for Training](#data-preprocessing-for-training)
3. [Training](#training)
4. [Checkpointing](#checkpointing)
5. [Some Training Results](#some-training-results)
6. [Exporting model and Testing](#exporting-model-and-testing)
7. [Running Code at CWRU HPC](#running-code-at-cwru-hpc)
8. [Acknowledgments](#acknowledgments)

## Getting Started

### Prerequisites

- Python 2.7 only supported
- PaddlePaddle the latest version

### Installation

1. You can use pip to install PaddlePaddle with a single command. But there are many little problems during installation and cost me a lot of time to fix(See the following Notes for detail).

```
sudo pip install paddlepaddle
```
- Note 1: Make sure that your default python version Python 2.7 series.
- Note 2: pip only supports manylinux1 standard, you’ll need to upgrade your pip to >9.0.0.
- Note 3: Use sudo pip instead or you’ll get permission denied error.

2. Make sure these libraries or tools installed: pkg-config, flac, ogg, vorbis, boost and swig,(I installed them via homebrew with proxy):
```
brew install pkg-config
brew install flac
brew install vorbis-tools
brew install boost
brew install swig
```

3. Run the setup script for the remaining dependencies.
```
git clone https://github.com/PaddlePaddle/DeepSpeech.git
cd DeepSpeech
sudo sh setup.sh
```
- Note : Remember to use “sudo” and using “brew install gcc” to install Fortran compiler.

## Data-Preprocessing for Training

### Generate Manifest

*DeepSpeech2 on PaddlePaddle* accepts a textual **manifest** file as its data set interface. A manifest file summarizes a set of speech data, with each line containing some meta data (e.g. filepath, transcription, duration) of one audio clip, in [JSON](http://www.json.org/) format, such as:

```
{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0001.flac", "duration": 3.275, "text": "stuff it into you his belly counselled him"}
{"audio_filepath": "/home/work/.cache/paddle/Libri/134686/1089-134686-0007.flac", "duration": 4.275, "text": "a cold lucid indifference reigned in his soul"}
```

To use your custom data, you only need to generate such manifest files to summarize the dataset. Given such summarized manifests, training, inference and all other modules can be aware of where to access the audio files, as well as their meta data including the transcription labels.

For how to generate such manifest files, please refer to `data/librispeech/librispeech.py`, which will download data and generate manifest files for LibriSpeech dataset.

### Compute Mean & Stddev for Normalizer

To perform z-score normalization (zero-mean, unit stddev) upon audio features, we have to estimate in advance the mean and standard deviation of the features, with some training samples:

```bash
python tools/compute_mean_std.py \
--num_samples 2000 \
--specgram_type linear \
--manifest_paths data/librispeech/manifest.train \
--output_path data/librispeech/mean_std.npz
```

It will compute the mean and standard deviation of power spectrum feature with 2000 random sampled audio clips listed in `data/librispeech/manifest.train` and save the results to `data/librispeech/mean_std.npz` for further usage.


### Build Vocabulary

A vocabulary of possible characters is required to convert the transcription into a list of token indices for training, and in decoding, to convert from a list of indices back to text again. Such a character-based vocabulary can be built with `tools/build_vocab.py`.

```bash
python tools/build_vocab.py \
--count_threshold 0 \
--vocab_path data/librispeech/eng_vocab.txt \
--manifest_paths data/librispeech/manifest.train
```

It will write a vocabuary file `data/librispeeech/eng_vocab.txt` with all transcription text in `data/librispeech/manifest.train`, without vocabulary truncation (`--count_threshold 0`).

### More Help

For more help on arguments:

```bash
python data/librispeech/librispeech.py --help
python tools/compute_mean_std.py --help
python tools/build_vocab.py --help
```

## Training

Language Model | Training Data | Token-based | Size | Descriptions
:-------------:| :------------:| :-----: | -----: | :-----------------
[Mandarin LM Small](http://cloud.dlnel.org/filepub/?uuid=d21861e4-4ed6-45bb-ad8e-ae417a43195e) | Baidu Internal Corpus | Char-based | 2.8 GB | Pruned with 0 1 2 4 4; <br/> About 0.13 billion n-grams; <br/> 'probing' binary with default settings
[Mandarin LM Large](http://cloud.dlnel.org/filepub/?uuid=245d02bb-cd01-4ebe-b079-b97be864ec37) | Baidu Internal Corpus | Char-based | 70.4 GB | No Pruning; <br/> About 3.7 billion n-grams; <br/> 'probing' binary with default settings

In this project, we download the 70.4 GB model using:
```
wget -O zhidao_giga.klm http://cloud.dlnel.org/filepub/?uuid=245d02bb-cd01-4ebe-b079-b97be864ec37
```

## CheckPointing
## Some Training Results
## Exporting model and Testing
## Running Code at CWRU HPC
1. Login
```
$ ssh sxx186@redhen1.case.edu
$ ssh sxx186@rider.case.edu
```

2. Require a computation node and load Singularity
```
$ srun -p gpu -C gpuk40 --gres=gpu:1 --pty bash
$ module load singularity/2.5.1
```
3. Download the Docker image
```
$ singularity pull docker://paddlepaddle/deep_speech:latest-gpu
$ git clone https://github.com/PaddlePaddle/DeepSpeech.git
```
4. Get into the image and reset $HOME
```
$ singularity shell -e -H /mnt/rds/redhen/gallina/Singularity/DeepSpeech2/DeepSpeech/deep_speech-latest-gpu.simg
$ unset HOME
$ export HOME="/mnt/rds/redhen/gallina/Singularity/DeepSpeech2/DeepSpeech/"
```
5. Run the code(run_data.sh as an example, and you can see Tiny data preparation done.)
```
$ cd DeepSpeech/examples/tiny/
$ sh run_data.sh
```

## Acknowledgments
* [Google Summer of Code 2018](https://summerofcode.withgoogle.com/)
* [Red Hen Lab](http://www.redhenlab.org/)
* [DeepSpeech2 on PaddlePaddle](https://github.com/PaddlePaddle/DeepSpeech)
* [DeepSpeech2 Paper](http://proceedings.mlr.press/v48/amodei16.pdf)
