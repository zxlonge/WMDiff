import os
import glob
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from PIL import Image


class CustomDatasetProcessor(object):
    """
    input : dataset.pkl
    """

    def __init__(self, data_file):
        self.data = pd.read_pickle(data_file)
        self.class_mapping = {
            0: 'center',
            1: 'donut',
            2: 'edge_Loc',
            3: 'edge_Ring',
            4: 'loc',
            5: 'near_full',
            6: 'random',
            7: 'scratch',
        }

    @staticmethod
    def save_image(arr, filepath='image.png', vmin=0, vmax=2):
        scaled_arr = (arr / vmax) * 255
        img = Image.fromarray(scaled_arr.astype(np.uint8))
        img.save(filepath, dpi=(500, 500))

    def write_images(self, root, class_mapping):
        os.makedirs(root, exist_ok=True)
        with tqdm(total=len(self.data), leave=True) as pbar:
            for i, row in self.data.iterrows():
                label = row['failureNum']
                class_dir = os.path.join(root, f'{class_mapping[label]}')
                os.makedirs(class_dir, exist_ok=True)
                pngfile = os.path.join(class_dir, f'{i:06}.png')
                self.save_image(row['waferMap'], pngfile)
                pbar.set_description_str(f" {root} - {i:06} ")
                pbar.update(1)


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description="Process Custom Dataset to individual image files.")
        parser.add_argument('--data_file_train', type=str, default='./dataset/wafermap/DiffusionAug.pkl')
        # parser.add_argument('--data_file_test', type=str, default='./dataset/wafermap/patternAdd_wafermap_test.pkl')
        parser.add_argument('--output_root', type=str, default='./data/wafermapAug')
        return parser.parse_args()


    args = parse_args()
    processor_train = CustomDatasetProcessor(data_file=args.data_file_train)
    processor_test = CustomDatasetProcessor(data_file=args.data_file_test)


    processor_train.write_images(root=os.path.join(args.output_root), class_mapping=processor_train.class_mapping)
    processor_test.write_images(root=os.path.join(args.output_root, 'test'), class_mapping=processor_test.class_mapping)


    print("Data processing complete.")
