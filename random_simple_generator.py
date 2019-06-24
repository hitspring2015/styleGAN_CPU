# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import config
import glob
import random

MODEL = './cache/2019-03-08-stylegan-animefaces-network-02051-021980.pkl'
PREFIX = 'Demo'
SEED = random.randint(0,10000)

def load_Gs(filepath):
    # Load pre-trained network.
    model_file = glob.glob(filepath)
    if len(model_file) == 1:
        model_file = open(model_file[0], "rb")
    else:
        raise Exception('Failed to find the model')

    _G, _D, Gs = pickle.load(model_file)

    # Print network details.
    #Gs.print_layers()

    return Gs

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Genetate Image
    os.makedirs(config.result_dir, exist_ok=True)
    save_name = PREFIX + '_' + str(random.getrandbits(64)) + '.png'
    save_path = os.path.join(config.result_dir, save_name)
    
    Gs = load_Gs(MODEL)
    rnd = np.random.RandomState(SEED)
    latents = rnd.randn(1, Gs.input_shape[1])
    
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    print('\n * Generating...'*6)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    print('\n * Done'*10)
    # Save image.
    PIL.Image.fromarray(images[0], 'RGB').save(save_path)
    print('* Image [' + save_name + '] has beed saved to ./result')

if __name__ == "__main__":
    main()
