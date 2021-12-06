# StarGAN-VC

This repository provides an official PyTorch implementation for [StarGAN-VC](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/stargan-vc2/index.html).

StarGAN-VC is a nonparallel many-to-many voice conversion (VC) method using star generative adversarial networks (StarGAN). The current version performs VC by first modifying the mel-spectrogram of input speech of an arbitrary speaker in accordance with a target speaker index, and then generating a waveform using the speaker-independent HifiGAN vocoder from the modified mel-spectrogram.

Audio samples are available [here](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/stargan-vc2/index.html).

## Papers

- [Hirokazu Kameoka](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/index-e.html), [Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/index.html), [Kou Tanaka](http://www.kecl.ntt.co.jp/people/tanaka.ko/index.html), and Nobukatsu Hojo, "**StarGAN-VC: Non-parallel many-to-many voice conversion using star generative adversarial networks**," in *Proc. 2018 IEEE Workshop on Spoken Language Technology ([SLT 2018](http://www.slt2018.org/))*, pp. 266-273, Dec. 2018. [**[Paper]**](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/publications/Kameoka2018SLT12_published.pdf) 
  
- [Hirokazu Kameoka](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/index-e.html), [Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/index.html), [Kou Tanaka](http://www.kecl.ntt.co.jp/people/tanaka.ko/index.html), and Nobukatsu Hojo, "**Nonparallel Voice Conversion With Augmented Classifier Star Generative Adversarial Networks**" *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 28, pp. 2982-2995, 2020. [**[Paper]**](https://ieeexplore.ieee.org/document/9256995) 

## Preparation

#### Requirements

- See `requirements.txt`.

#### Dataset

1. Setup your training and test sets. The data structure should look like:


```bash
/path/to/dataset/training
├── spk_1
│   ├── utt1.wav
│   ...
├── spk_2
│   ├── utt1.wav
│   ...
└── spk_N
    ├── utt1.wav
    ...
    
/path/to/dataset/test
├── spk_1
│   ├── utt1.wav
│   ...
├── spk_2
│   ├── utt1.wav
│   ...
└── spk_N
    ├── utt1.wav
    ...
```

#### Waveform generator

1. Place a copy of the directory `parallel_wavegan` from https://github.com/kan-bayashi/ParallelWaveGAN in `hifigan/` (or `pwg/`).
2. HifiGAN models trained on several databases can be found [here](https://drive.google.com/drive/folders/1RvagKsKaCih0qhRP6XkSF07r3uNFhB5T?usp=sharing). Once these are downloaded, place them in `hifigan/egs/`. Please contact me if you have any problems downloading.
3. Optionally, Parallel WaveGAN can be used instead for waveform generation. The trained models are available [here](https://drive.google.com/drive/folders/1zRYZ9dx16dONn1SEuO4wXjjgJHaYSKwb?usp=sharing). Once these are downloaded, place them in `pwg/egs/`. 

## Main

#### Train

To run all stages for model training, execute:

```bash
./recipes/run_train.sh [-g gpu] [-a arch_type] [-l loss_type] [-s stage] [-e exp_name]
```

- Options:

  ```bash
  -g: GPU device (default: -1)
  #    -1 indicates CPU
  -a: Generator architecture type ("conv" or "rnn")
  #    conv: 1D fully convolutional network (default)
  #    rnn: Bidirectional long short-term memory network
  -l: Loss type ("cgan", "wgan", or "lsgan")
  #    cgan: Cross-entropy GAN
  #    wgan: Wasserstein GAN with the gradient penalty loss (default)
  #    lsgan: Least squares GAN
  -s: Stage to start (0 or 1)
  #    Stages 0 and 1 correspond to feature extraction and model training, respectively.
  -e: Experiment name (default: "conv_wgan_exp1")
  #    This name will be used at test time to specify which trained model to load.
  ```

- Examples:

  ```bash
  # To run the training from scratch with the default settings:
  ./recipes/run_train.sh
  
  # To skip the feature extraction stage:
  ./recipes/run_train.sh -s 1
  
  # To set the gpu device to, say, 0:
  ./recipes/run_train.sh -g 0
  
  # To use a generator with a recurrent architecture:
  ./recipes/run_train.sh -a rnn -e rnn_wgan_exp1
  
  # To use the cross-entropy adversarial loss:
  ./recipes/run_train.sh -l cgan -e conv_cgan_exp1
  
  # To use the least-squares adversarial loss:
  ./recipes/run_train.sh -l lsgan -e conv_lsgan_exp1
  ```

See other scripts in `recipes` for examples of training on different datasets. 

To monitor the training process, use tensorboard:

```bash
tensorboard [--logdir log_path]
```

#### Test

To perform conversion, execute:

```bash
./recipes/run_test.sh [-g gpu] [-e exp_name] [-c checkpoint] [-v vocoder]
```

- Options:

  ```bash
  -g: GPU device (default: -1)
  #    -1 indicates CPU
  -e: Experiment name (e.g., "conv_wgan_exp1")
  -c: Model checkpoint to load (default: 0)
  #    0 indicates the newest model
  -v: Vocoder type ("hifigan" or "pwg")
  #    hifigan: HifiGAN (default)
  #    pwg: Parallel WaveGAN
  ```

- Examples:

  ```bash
  # To perform conversion with the default settings:
  ./recipes/run_test.sh -g 0 -e conv_wgan_exp1
  
  # To use Parallel WaveGAN as an alternative for waveform generation:
  ./recipes/run_test.sh -g 0 -e conv_wgan_exp1 -v pwg
  ```

## Citation

If you find this work useful for your research, please cite our papers.

```
@INPROCEEDINGS{Kameoka2018SLT_StarGAN-VC,
  author={Hirokazu Kameoka and Takuhiro Kaneko and Kou Tanaka and Nobukatsu Hojo},
  booktitle={Proc. 2018 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={StarGAN-VC: Non-parallel Many-to-Many Voice Conversion Using Star Generative Adversarial Networks}, 
  year={2018},
  pages={266--273}}
@Article{Kameoka2020IEEETrans_StarGAN-VC,
  author={Hirokazu Kameoka and Takuhiro Kaneko and Kou Tanaka and Nobukatsu Hojo},
  title={Nonparallel Voice Conversion With Augmented Classifier Star Generative Adversarial Networks},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={28},
  pages={2982--2995},
  year={2020}}
```

## Author

Hirokazu Kameoka ([@kamepong](https://github.com/kamepong))

E-mail: kame.hirokazu@gmail.com
