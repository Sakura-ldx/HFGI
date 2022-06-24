import os

import dlib
import numpy as np
import scipy
import torch
from PIL import Image

from configs import data_configs
from utils.common import tensor2im
from utils.model_utils import setup_model


def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(filepath, predictor):
    """
    :param filepath: str
    :return: PIL Image
    """

    lm = get_landmark(filepath, predictor)

    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = Image.open(filepath)

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]

    # Transform.
    img = img.transform(
        (transform_size, transform_size),
        Image.QUAD,
        (quad + 0.5).flatten(),
        Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    # Return aligned image.
    return img


class HFGI:

    def __init__(self,
                 checkpoint_path: str = './checkpoint/ckpt.pt',
                 # out_dir: str = './out/inversion',
                 output_resize: int = 256,
                 device='cuda'):

        net, opts = setup_model(checkpoint_path, device)
        # opts['exp_dir'] = out_dir
        # opts['output_resize'] = output_resize

        dataset_args = data_configs.DATASETS[opts.dataset_type]
        transforms_dict = dataset_args['transforms'](opts).get_transforms()

        self.output_size = output_resize
        self.device = device
        self.is_cars = 'car' in opts.dataset_type
        self.opts = opts
        self.net = net
        # self.latent_editor = LatentEditor(self.net.decoder, is_cars=self.is_cars)
        self.transform = transforms_dict['transform_inference']
        self.predictor = dlib.shape_predictor("./checkpoint/shape_predictor_68_face_landmarks.dat")

    def get_image(self, image_path, align=True):
        if align:
            from_im = align_face(filepath=image_path, predictor=self.predictor)
            # print(type(from_im))
            from_im = from_im.convert('RGB')
            # print(from_im.shape)
        else:
            from_im = Image.open(image_path).convert('RGB')

        from_im = self.transform(from_im).unsqueeze(0)
        return from_im

    def get_images(self, image_paths, align=True):
        images = []
        for image_path in image_paths:
            image = self.get_image(image_path, align)
            images.append(image)
        return torch.cat(images).to(self.device).float()

    def get_latents(self, images):
        latents = self.net.encoder(images)
        if self.net.opts.start_from_latent_avg:
            if latents.ndim == 2:
                latents = latents + self.net.latent_avg.repeat(latents.shape[0], 1, 1)[:, 0, :]
            else:
                latents = latents + self.net.latent_avg.repeat(latents.shape[0], 1, 1)
        if latents.shape[1] == 18 and self.is_cars:
            latents = latents[:, :16, :]
        return latents

    def save_image(self, image, save_dir, name):
        result = tensor2im(image)
        im_save_path = os.path.join(save_dir, f"{name}.jpg")
        Image.fromarray(np.array(result)).save(im_save_path)

    def save_images(self, images, save_dir, names):
        os.makedirs(save_dir, exist_ok=True)
        for i, name in enumerate(names):
            self.save_image(images[i], save_dir, name)

    def inversion(self, image_paths, save_dir='./out/inversion', names=None, align=True):
        input_images = self.get_images(image_paths, align)
        with torch.no_grad():
            # first stage
            latents = self.get_latents(input_images)
            images_init, _ = self.net.decoder([latents], None, input_is_latent=True,
                                              randomize_noise=False, return_latents=True)

            # second stage
            images_init = torch.nn.functional.interpolate(torch.clamp(images_init, -1., 1.), size=(256, 256), mode='bilinear')
            res = input_images - images_init
            res_aligns = self.net.grid_align(torch.cat((res, images_init), 1))
            conditions = self.net.residue(res_aligns)
            images_final, _ = self.net.decoder([latents], conditions, input_is_latent=True,
                                               randomize_noise=False, return_latents=True)
            if self.is_cars:
                images_final = images_final[:, :, 64:448, :]

            if self.output_size != self.opts.stylegan_size:
                images_final = torch.nn.functional.interpolate(images_final,
                                                               size=(self.output_size, self.output_size),
                                                               mode='bilinear')
        if not names:
            names = [os.path.basename(image_path)[0:-4] for image_path in image_paths]
        self.save_images(images_final, save_dir, names)

    def edit(self, image_paths, direction_path, factor=1.0, save_dir='./out/edit', names=None, align=True):
        input_images = self.get_images(image_paths, align)
        with torch.no_grad():
            # first stage
            latents = self.get_latents(input_images)
            direction = torch.load(direction_path, map_location=self.device)
            latents_edit = latents + factor * direction
            images_init, _ = self.net.decoder([latents_edit], None, input_is_latent=True,
                                              randomize_noise=False, return_latents=True)

            # second stage
            images_init = torch.nn.functional.interpolate(torch.clamp(images_init, -1., 1.), size=(256, 256), mode='bilinear')
            res = input_images - images_init
            res_aligns = self.net.grid_align(torch.cat((res, images_init), 1))
            conditions = self.net.residue(res_aligns)
            images_final, _ = self.net.decoder([latents_edit], conditions, input_is_latent=True,
                                               randomize_noise=False, return_latents=True)
            if self.is_cars:
                images_final = images_final[:, :, 64:448, :]

            if self.output_size != self.opts.stylegan_size:
                images_final = torch.nn.functional.interpolate(images_final,
                                                               size=(self.output_size, self.output_size),
                                                               mode='bilinear')
        if not names:
            names = [os.path.basename(image_path)[0:-4] + '_' + os.path.basename(direction_path)[0:-3] for image_path in image_paths]
        self.save_images(images_final, save_dir, names)

    def mix(self, image_paths1, image_paths2, factor=0.5, save_dir='./out/mix', names=None, align=True):
        assert len(image_paths1) == len(image_paths2), "number of images is not same"
        input_images1 = self.get_images(image_paths1, align)
        input_images2 = self.get_images(image_paths2, align)
        with torch.no_grad():
            # first stage
            latents1 = self.get_latents(input_images1)
            latents2 = self.get_latents(input_images2)
            latents_mix = factor * latents1 + (1 - factor) * latents2
            images_mix, _ = self.net.decoder([latents_mix], None, input_is_latent=True,
                                             randomize_noise=False, return_latents=True)

            if self.is_cars:
                images_mix = images_mix[:, :, 64:448, :]

            if self.output_size != self.opts.stylegan_size:
                images_mix = torch.nn.functional.interpolate(images_mix,
                                                             size=(self.output_size, self.output_size),
                                                             mode='bilinear')
        if not names:
            names = [os.path.basename(image_paths1[i])[0:-4] + '_' + os.path.basename(image_paths2[i])[0:-4] for i in range(len(image_paths1))]
        self.save_images(images_mix, save_dir, names)


if __name__ == '__main__':
    hfgi = HFGI()
    hfgi.inversion(['./test_imgs/00051.jpg'], align=False)
    hfgi.edit(['./test_imgs/00051.jpg'], direction_path='./editings/interfacegan_directions/age.pt', align=False)
    hfgi.mix(['./test_imgs/00051.jpg'], ['./test_imgs/00511.jpg'], align=False)
