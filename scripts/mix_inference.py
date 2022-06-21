import argparse
import torch
import numpy as np
import sys
import os

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import MixDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from PIL import Image
from editings import latent_editor


def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'car' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    aligner = net.grid_align
    args, data_loader = setup_data_loader(args, opts)

    # initial inversion
    latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)

    os.makedirs(args.save_dir, exist_ok=True)

    # perform high-fidelity inversion or editing
    for i, batch in enumerate(data_loader):
        if args.n_sample is not None and i > args.n_sample:
            print('inference finished!')
            break
        x1, x2 = batch
        x1, x2 = x1.to(device).float(), x2.to(device).float()

        # calculate the distortion map
        latents1, latents2 = latent_codes[i][0], latent_codes[i][1]
        latents_mix = (args.mix_degree * latents1 + (1 - args.mix_degree) * latents2).to(device)
        imgs, _ = generator([latents_mix], None, input_is_latent=True,
                            randomize_noise=False, return_latents=True)
        imgs = torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256, 256), mode='bilinear')
        res1 = x1 - imgs
        res2 = x2 - imgs

        # align the distortion map
        res_align1 = net.grid_align(torch.cat((res1, imgs), 1))
        res_align2 = net.grid_align(torch.cat((res2, imgs), 1))

        # consultation fusion
        conditions1 = net.residue(res_align1)
        conditions2 = net.residue(res_align2)
        conditions_mix = get_conditions_mix(conditions1, conditions2)
        imgs, _ = generator([latents_mix], conditions_mix, input_is_latent=True, randomize_noise=False,
                            return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]

        # save images
        imgs = torch.nn.functional.interpolate(imgs, size=(256, 256), mode='bilinear')
        save_image(imgs, args.save_dir, i)


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = [os.path.join(args.images_dir, d) for d in os.listdir(args.images_dir)]
    print(f"images path: {images_path}")
    align_function = None
    test_dataset = MixDataset(root1=images_path[0],
                              root2=images_path[1],
                              transform=transforms_dict['transform_test'],
                              preprocess=align_function,
                              opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=0,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            image1, image2 = batch
            image1, image2 = image1.to(device).float(), image2.to(device).float()
            latents1 = get_latents(net, image1, is_cars)
            latents2 = get_latents(net, image2, is_cars)
            all_latents.append([latents1, latents2])
            i += len(latents1)
    return all_latents


def get_conditions_mix(conditions1, conditions2):
    conditions_mix = []
    conditions_mix.append(conditions1[0] + conditions2[0] + conditions1[0] * conditions2[0])
    conditions_mix.append(conditions1[1] + conditions2[1])
    return conditions_mix


def save_image(imgs, save_dir, idx):
    i = 0
    for img in imgs:
        result = tensor2im(img)
        im_save_path = os.path.join(save_dir, f"{idx:05d}_{i}.jpg")
        Image.fromarray(np.array(result)).save(im_save_path)
        i += 1


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None, help="The directory to the images")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--mix_degree", type=float, default=0.5, help="edit degreee")
    parser.add_argument("--ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)
