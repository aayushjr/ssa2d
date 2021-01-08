import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import imgaug


def display_one_hot(one_hot, original, threshold, is_categorical, class_names, should_save=False,
                    out_folder_path='', name='out_img'):
    segmentation = (original * 255).astype('uint8')

    h, w, = original.shape[:2]
    # resize one-hot
    resizer = imgaug.augmenters.Resize({'height': h, 'width': w}, interpolation='nearest')
    # one_hot = np.transpose(one_hot, [2, 0, 1])
    one_hot = resizer(image=one_hot)
    # one_hot = np.transpose(one_hot, [1, 2, 0])

    class_ids = set()
    for row in range(h):
        for col in range(w):
            if is_categorical:
                class_id = int(np.argmax(one_hot[row, col]))
                segmentation[row, col] = (np.asarray(colors[class_id]) + segmentation[row, col]) / 2
                class_ids.add(class_id)
            else:
                for class_id in range(len(class_names)):
                    if one_hot[row, col, class_id] < threshold:
                        continue

                    segmentation[row, col] = (np.asarray(colors[class_id]) + segmentation[row, col]) / 2
                    class_ids.add(class_id)

    labels = []
    for class_id in class_ids:
        rgb = np.asarray(colors[class_id], dtype=float) / 255
        labels.append(mpatches.Patch(color=rgb, label=class_names[class_id]))

    display_img(segmentation, should_save, out_folder_path, name, labels=labels)


def display_binary(onehot, draw_on=None, should_save=False, out_folder_path='', name='out_img', class_names=None,
                   threshold=0.1):
    resizer = imgaug.augmenters.Resize({'height': draw_on.shape[0], 'width': draw_on.shape[1]}, interpolation=0)
    segmented = np.copy(draw_on)
    onehot = resizer(np.transpose(onehot, [2, 0, 1]))
    onehot = np.transpose(onehot, [1, 2, 0])
    class_ids = set()
    labels = []

    for row in range(onehot.shape[0]):
        for col in range(onehot.shape[1]):
            for class_id in range(len(class_names)):
                if onehot[row, col, class_id] < threshold:
                    # segmented[row, col] = (0, 0, 0)
                    continue

                segmented[row, col] = ((np.asarray(colors[class_id]) + segmented[row, col]) / 2)
                class_ids.add(class_id)

    for class_id in class_ids:
        rgb = np.asarray(colors[class_id], dtype=float) / 255
        labels.append(mpatches.Patch(color=rgb, label=class_names[class_id]))

    display_img(segmented, should_save, out_folder_path, name, labels=labels)


def display_img(image, should_save=False, out_folder_path='', name='out_img', labels=[]):
    # plt.imshow(np.asarray(image, 'uint8'))

    plt.imshow(image)

    if len(labels) != 0:
        plt.legend(handles=labels)
    if should_save:
        plt.savefig(construct_out_path(out_folder_path, name))

    plt.show()


def construct_out_path(out_folder_path='', name='out_img', extension='.png', create_folder=True):
    if create_folder and not os.path.exists(out_folder_path):
        os.mkdir(out_folder_path)

    return os.path.join(out_folder_path, name + extension)


def display_centroids(centroids, threshold, should_save=False, out_folder_path='', name='out_img'):
    # display centroids
    centroids[centroids < threshold] = 0
    centroids = np.asarray(centroids * 255, dtype='uint8')
    display_img(centroids.squeeze(-1), should_save, out_folder_path, name)


colors = [(178, 0, 0), (89, 0, 0), (153, 38, 38), (178, 89, 89), (217, 163, 163), (64, 48, 48),
          (255, 145, 128), (115, 65, 57), (127, 34, 0), (191, 86, 48), (51, 23, 13), (153, 97, 77),
          (255, 208, 191), (242, 97, 0), (76, 31, 0), (191, 105, 48), (242, 170, 121), (76, 54, 38),
          (128, 108, 96), (255, 166, 64), (115, 75, 29), (191, 147, 96), (217, 191, 163), (140, 94, 0),
          (255, 191, 64), (51, 38, 13), (127, 106, 64), (51, 47, 38), (178, 143, 0), (89, 71, 0),
          (255, 230, 128), (217, 206, 163), (238, 255, 0), (194, 204, 51), (109, 115, 29), (138, 140, 105),
          (119, 179, 0), (51, 77, 0), (34, 51, 0), (138, 166, 83), (97, 242, 0), (217, 255, 191),
          (73, 89, 67), (40, 115, 29), (144, 255, 128), (0, 153, 0), (115, 153, 120), (26, 51, 32),
          (0, 77, 31), (32, 128, 83), (163, 217, 191), (0, 191, 128), (0, 255, 204), (0, 89, 71),
          (64, 255, 242), (77, 153, 148), (32, 64, 62), (0, 102, 128), (64, 217, 255), (153, 194, 204),
          (0, 128, 191), (0, 34, 51), (19, 57, 77), (102, 170, 204), (67, 82, 89), (0, 102, 191),
          (32, 83, 128), (105, 124, 140), (0, 92, 230), (0, 51, 128), (0, 31, 77), (128, 179, 255),
          (191, 217, 255), (0, 0, 77), (54, 38, 153), (200, 191, 255), (50, 48, 64), (34, 0, 128),
          (137, 108, 217), (104, 96, 128), (92, 0, 230), (98, 70, 140), (166, 64, 255), (51, 0, 64),
          (130, 38, 153), (206, 163, 217), (238, 0, 255), (255, 128, 246), (166, 83, 160), (204, 0, 163),
          (115, 0, 92), (64, 32, 57), (102, 77, 94), (217, 0, 116), (76, 0, 41), (153, 38, 99),
          (255, 128, 196), (204, 153, 180), (166, 41, 75), (217, 108, 137), (115, 57, 73), (255, 0, 34),
          (255, 64, 89), (115, 29, 40), (76, 19, 27), (140, 105, 110)]
