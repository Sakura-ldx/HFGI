from argparse import ArgumentParser

from configs.paths_config import model_paths


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # arguments for inference script
        self.parser.add_argument('--out_dir', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to HyperStyle model checkpoint')
        self.parser.add_argument('--resize_outputs', action='store_true',
                                 help='Whether to resize outputs to 256x256 or keep at original output resolution')

        # arguments for loading pre-trained encoder
        self.parser.add_argument('--load_w_encoder', action='store_true', help='Whether to load the w e4e encoder.')
        self.parser.add_argument('--w_encoder_checkpoint_path', default=model_paths["faces_w_encoder"], type=str,
                                 help='Path to pre-trained W-encoder.')
        self.parser.add_argument('--w_encoder_type', default='WEncoder',
                                 help='Encoder type for the encoder used to get the initial inversion')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
