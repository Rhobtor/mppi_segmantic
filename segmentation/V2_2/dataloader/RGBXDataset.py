from operator import gt
import os
from pickletools import uint8
import cv2
import torch
import numpy as np

import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._gt_path = setting['gt_root']
        self._gt_format = setting['gt_format']
        self._transform_gt = setting['transform_gt']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
        x_path = os.path.join(self._x_path, item_name + self._x_format)
        gt_path = os.path.join(self._gt_path, item_name + self._gt_format)
        # Check the following settings if necessary
        try:
            rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)
            gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
            if self._transform_gt:
                gt = self._gt_transform(gt)
            if self._x_single_channel:
                x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
                x = cv2.merge([x, x, x])
            else:
                x =  self._open_image(x_path, cv2.COLOR_BGR2RGB)
            if self.preprocess is not None:
                rgb, gt, x = self.preprocess(rgb, gt, x)
            if self._split_name == 'train':
                rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
                gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
                x = torch.from_numpy(np.ascontiguousarray(x)).float()
        
        except FileNotFoundError as e:
            logger.warning(f"File not found at index {index}: {e}")
            rgb = np.zeros((config.image_height, config.image_width, 3), dtype=np.uint8)
            gt_path = np.zeros((config.image_height, config.image_width), dtype=np.uint8)
            x = np.zeros((config.image_height, config.image_width, 3), dtype=np.uint8)
        
        output_dict = dict(data=rgb, label=gt, modal_x=x, fn=str(item_name), n=len(self._file_names))
        
        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val']
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip().zfill(6)
            file_names.append(file_name)
        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)                          
        new_file_names = self._file_names * (length // files_len)   

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        if filepath.endswith('.npy'):
        # Cargar archivo .npy usando numpy
            img = np.load(filepath).astype(dtype)
        else:
        # Cargar archivo de imagen usando OpenCV
            img = cv2.imread(filepath, mode)
            if img is not None:
                img = img.astype(dtype)
            else:
                raise FileNotFoundError(f"File not found or could not be opened: {filepath}")
        return img
        

    @staticmethod
    def _gt_transform(gt):
        """
        Remapea las etiquetas originales del dataset:

            Original 1..12 -> entrenamiento 0..11
            Original 0     -> 255 (unlabeled / ignore)
            Original 255   -> 255 (ignore)
    """

        # Primero comprobamos que la máscara solo contenga
        # los IDs que realmente existen en el dataset.
        unique_ids = np.unique(gt)

        allowed_ids = set(range(13)) | {255}
        unknown_ids = [
            int(x) for x in unique_ids
            if int(x) not in allowed_ids
        ]

        if unknown_ids:
            raise ValueError(
                f"La máscara contiene IDs desconocidos: {unknown_ids}"
            )

        # Inicialmente todo se considera ignore.
        transformed_gt = np.full(
            gt.shape,
            255,
            dtype=np.uint8
        )

        # Clases válidas:
        # 1..12 -> 0..11
        valid_pixels = (gt >= 1) & (gt <= 12)

        transformed_gt[valid_pixels] = (
            gt[valid_pixels] - 1
        )

        # gt == 0   permanece como 255
        # gt == 255 permanece como 255

        return transformed_gt

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors
