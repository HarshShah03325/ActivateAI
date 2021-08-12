# ActivateAI
The Real-Time assisted detection system provides the ability to detect the trigger word 'activate'.
It is the technology that allows devices like Amazon Alexa and Google Home to wake up upon hearing a certain word.
It is a Natural Language Processing Model developed with advanced Recurrent Neural Networks.
Everytime it hears you say the word 'activate', it will produce a chiming sound.

<p align="center">
  <img width="460" height="300" src="/assets/logo.jpg">
</p>


## Table of Contents

- [Generation of Data](#generation-of-data)
- [Data preprocessing](#preprocessing-the-data)
- [Recurrent Neural Network Model](#recurrent-neural-network-model)
- [Training and development](#training-and-development)

- [Demo](#demo)
- [References](#references)

## 🤖 Technology Stack
- Framewoks: Keras, Tensorflow


## [Generation of Data](#sections)

- 3 types of audio recordings are present
  - Positives
  - Negatives
  - Backgrounds
- Positives are the audios which have the trigger word 'activate' that has to detected.
- Negatives are the audios which contain words other than the trigger word.
- Backgrounds are the audios which contain random noises
- To generate training example, we insert positives and negatives into the backgrounds in a non-overlapping condition.

## [Data preprocessing](#sections)
- A microphone records little variations in air pressure over time, and it is these little variations in air pressure that your ear also perceives as sound. You can think of an audio recording is a long list of numbers measuring the little air pressure changes detected by the microphone. 
- It is quite difficult to figure out from this "raw" representation of audio whether the word "activate" was said. In order to help our sequence model more easily learn to detect triggerwords, we will compute a spectrogram of the audio.
- Visual representation of frequencies of a given signal with time is called Spectrogram. In a spectrogram representation plot one axis represents the time, the second axis represents frequencies and the colors represent magnitude (amplitude) of the observed frequency at a particular time.
- We compute the following spectogram from our training example.

<p align="center">
  <img width="460" height="300" src="/assets/spectogram.png">
</p>



## [Recurrent Neural Network model](#sections)

- The architecture of the model consists of 1-D convolutional layers, GRU layers, and dense layers.
- The bottom most layer is a 1D Convolution layer.It converts the input of length 5511 timestamps into 1375 output timestamps.
- Convolution layer is followed by batch normalization, activation and a drop-out layer.
- GRUs(Gated recurrent units) are improved version of standard recurrent neural network. GRU aims to solve the vanishing gradient problem which comes with a standard recurrent neural network.
- A unidirectional RNN is used rather than a bi-directional RNN, since we want to detect the trigger word immediately after its said.

<p align="center">
  <img width="460" height="300" src="/assets/model.png">
</p>

## [Training and development](#sections)
- The model has about 50,000 trainable parameters. The model is trained on a large training set of 4000 examples generated.
- Adam optimizer and binary_crossentropy loss function were used for training.

## [Project Setup](#sections)

```
git clone
```



## [Demo]





## [References]
- This implementation was inspired by the Deep Learning Specialization on Coursera by _Andrew Ng_.
- [Trigger Word Detection - Coursera](https://www.coursera.org/learn/nlp-sequence-models/lecture/Li4ts/trigger-word-detection)




