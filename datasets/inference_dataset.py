from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None, preprocess=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.preprocess = preprocess
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		if self.preprocess is not None:
			from_im = self.preprocess(from_path)
		else:
			from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im


class MixDataset(Dataset):

	def __init__(self, root1, root2, opts, transform=None, preprocess=None):
		self.paths1 = sorted(data_utils.make_dataset(root1))
		self.paths2 = sorted(data_utils.make_dataset(root2))
		self.transform = transform
		self.preprocess = preprocess
		self.opts = opts

	def __len__(self):
		return min(len(self.paths1), len(self.paths2))

	def __getitem__(self, index):
		image1_path = self.paths1[index]
		image2_path = self.paths2[index]
		if self.preprocess is not None:
			image1 = self.preprocess(image1_path)
			image2 = self.preprocess(image2_path)
		else:
			image1 = Image.open(image1_path).convert('RGB')
			image2 = Image.open(image2_path).convert('RGB')
		if self.transform:
			image1 = self.transform(image1)
			image2 = self.transform(image2)
		return image1, image2
