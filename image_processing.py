from PIL import Image
import pandas as pd
from collections import Counter
import numpy as np
import os


class Image_Processing:

    def __init__(self, path=''):
        self.path = path

    def gray_scale_conver(self, input_image, is_save=True):  # input: PIL image

        gray_image = input_image.convert('L')
        gray_pixels = gray_image.load()
        width, height = gray_image.size

        gray_list = np.empty((height, width), dtype=np.uint8)
        gray_threshold = 130

        # def is_inverse():

        for y in range(height):
            for x in range(width):
                if gray_pixels[x, y] < gray_threshold:
                    gray_list[y, x] = 0
                else:
                    gray_list[y, x] = 255

        grayed_image = Image.fromarray(gray_list, mode='L')

        if is_save:
            output_file_path = os.path.join(self.path, 'gray_image.jpg')
            grayed_image.save(output_file_path)

        return grayed_image, gray_list

    def split_image(self, input_grayed_list, is_save=True, num=''):  # input: numpy, uint8, gray image
        # output : list[numpy, uint8, gray image]
        outcome_images_list = []
        split_row = set()

        for row in range(1, input_grayed_list.shape[0]):
            upper = len(np.unique(input_grayed_list[row - 1]))
            lower = len(np.unique(input_grayed_list[row]))
            if upper != lower:
                if upper < lower:
                    split_row.add(row - 1)
                else:
                    split_row.add(row)

        split_row = sorted(split_row)

        up_bound = 0
        end_bound = split_row[0]

        for ele in range(1, len(split_row)):
            split_col = set()
            up_bound = end_bound
            end_bound = split_row[ele]
            if len(np.unique(input_grayed_list[up_bound: end_bound, :])) != 1:
                for col in range(1, input_grayed_list.shape[1]):
                    left_bound = len(np.unique(input_grayed_list[up_bound: end_bound:, col - 1]))
                    right_bound = len(np.unique(input_grayed_list[up_bound: end_bound:, col]))
                    if left_bound != right_bound:
                        if left_bound < right_bound:
                            split_col.add(col - 1)
                        else:
                            split_col.add(col)

                split_col = sorted(split_col)

                target_list = input_grayed_list[up_bound: end_bound, split_col[0]: split_col[-1]]
                output_list = np.full((target_list.shape[0] + 10, target_list.shape[1] + 10), 255,
                                      dtype=np.uint8)  # only np.unit8 type

                for i in range(target_list.shape[0]):
                    for j in range(target_list.shape[1]):
                        output_list[i + 5, j + 5] = target_list[i, j]

                new_image = Image.fromarray(output_list, mode='L')
                outcome_images_list.append(output_list)

                if is_save:
                    if 'test_images' in self.path:
                        output_file_path = os.path.join(self.path, f'splited_image_{ele}.jpg')
                    else:
                        output_file_path = os.path.join(self.path, f'image_{num}.jpg')
                    new_image.save(output_file_path)

        return outcome_images_list


if __name__ == '__main__':
    abs_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(abs_file_path, 'data_set', 'test_images')
    test_image_path = os.path.join(file_path, 'image_test_1.png')

    input_image = Image.open(test_image_path)

    image_processed = Image_Processing(file_path)

    grayed_image_list = image_processed.gray_scale_conver(input_image)[1]

    image_processed.split_image(grayed_image_list)

    pass
