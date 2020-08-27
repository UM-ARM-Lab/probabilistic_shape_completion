#!/usr/bin/env bash

./train.py --group Flow
./train.py --group FlowYCB
./train.py --group NormalizingAE
./train.py --group VAE
./train.py --group VAE_GAN
./train.py --group 3D_rec_gan
./train.py --group NormalizingAE_YCB
./train.py --group VAE_YCB
./train.py --group VAE_GAN_YCB
./train.py --group 3D_rec_gan_YCB
./train.py --group NormalizingAE_YCB_noise
./train.py --group NormalizingAE_noise
