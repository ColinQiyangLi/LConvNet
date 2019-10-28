import os

import numpy as np
from imageio import imsave

from lconvnet.tasks.gan.gan_utils import process_config
from lconvnet.tasks.dualnets.distrib import GANSampler
from lconvnet.tasks.gan.data_loader.data_loader import dataloader


def collect_images(sampler, num_imgs, im_size, num_channels, sample_size):
    # Collect images.
    sampled_images = np.zeros((num_imgs, im_size, im_size, num_channels))

    count = 0
    while count < num_imgs - sample_size:
        curr_imgs = sampler(cfg.distrib1.sample_size).detach().cpu().numpy().transpose((0, 2, 3, 1))
        assert curr_imgs.shape[0] == sample_size, "Doens't match sample size, count: {}".format(count)
        sampled_images[count:count + sample_size] = curr_imgs

        count += sample_size

    # Add the last bit.
    last_samples = gan_sampler(cfg.distrib1.sample_size).detach().cpu().numpy().transpose((0, 2, 3, 1))
    sampled_images[count:] = last_samples[num_imgs - count]

    return sampled_images


def save_images(imgs, path):
    imgs = transform_imgs(imgs)

    for i in range(imgs.shape[0]):
        if i % 10000 == 0:
            print("Saved {} images. ".format(i))
        curr_im = imgs[i]
        curr_path = os.path.join(path, "im_{}.png".format(i))
        imsave(curr_path, curr_im)


def transform_imgs(imgs):
    imgs = (imgs + 1) / 2
    assert imgs.min() >= 0.0, "img min is: {}".format(imgs.min())
    assert imgs.max() <= 1.0, "img max is: {}".format(imgs.max())

    imgs = (255*imgs).astype(np.uint8)

    return imgs


if __name__ == "__main__":
    # Parse the config.
    cfg = process_config()

    # Quick checks.
    assert cfg.distrib1.generate_type == "generated"

    print("Working on generated samples. ")

    # Load the gan loader.
    gan_sampler = GANSampler(cfg.distrib1, "generated")

    # Collect images.
    generated_images = collect_images(gan_sampler, cfg.num_imgs, cfg.im_size, cfg.num_channels, cfg.distrib1.sample_size)

    # Save the generated images.
    generated_path = os.path.join(cfg.base_save_path, "generated")
    os.makedirs(generated_path, exist_ok=True)
    save_images(generated_images, generated_path)

    print("Working on real samples. ")

    # Save the real samples.
    dataset_path = os.path.join(cfg.base_save_path, "dataset")
    os.makedirs(dataset_path, exist_ok=True)
    loader = dataloader(cfg.dataset, cfg.im_size, cfg.dataset_size, dataset_path)
    real_images = loader.__iter__().__next__()[0].detach().cpu().numpy().transpose((0, 2, 3, 1))

    real_path = os.path.join(cfg.base_save_path, "real")
    os.makedirs(real_path, exist_ok=True)
    save_images(real_images, real_path)

