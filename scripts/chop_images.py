import os
import numpy as np
from PIL import Image

data_dir = '/data/buildings/AerialImageDataset/train'
n = len(os.listdir(os.path.join(data_dir, 'gt')))
print('starting cropping {} images'.format(n))

for i, img_path in enumerate(os.listdir(os.path.join(data_dir, 'gt'))):
    base_img_path = img_path.replace('.tif', '')
    gt = Image.open(os.path.join(data_dir, 'gt', img_path))
    im = Image.open(os.path.join(data_dir, 'images', img_path))
    imarray = np.array(im)
    gtarray = np.array(gt)
    for h in range(10):
        for w in range(10):
            tmp_dict = {}
            min_h = 500 * h
            max_h = 500 * (h + 1)
            min_w = 500 * w
            max_w = 500 * (w + 1)
            new_imgarray = imarray[min_h:max_h, min_w:max_w]
            new_gtarray = gtarray[min_h:max_h, min_w:max_w]
            tmp_dict["id"] = h * 10 + w
            tmp_dict["license"] = 0
            tmp_dict["coco_url"] = 'none'
            tmp_dict["flickr_url"] = 'none'
            tmp_dict["width"] = new_imgarray.shape[1]
            tmp_dict["height"] = new_imgarray.shape[0]
            tmp_dict["file_name"] = os.path.join(data_dir, 'cropped', base_img_path + '_{}.png'.format(h * 10 + w))
            tmp_dict["annotation_name"] = os.path.join(data_dir, 'annotations',
                                                       base_img_path + '_{}.png'.format(h * 10 + w))
            tmp_dict["date_captured"] = 20200720
            # print(np.unique(new_imgarray, return_counts=True))
            if np.sum(new_gtarray) != 0:
                Image.fromarray(new_imgarray).save(tmp_dict['file_name'])
                Image.fromarray(new_gtarray).save(tmp_dict['annotation_name'])
            # images.append(tmp_dict)
    if i % 10 == 0:
        print('processed: {}/{}'.format(i, n))
n = len(os.listdir(os.path.join(data_dir, 'annotations')))
m = len(os.listdir(os.path.join(data_dir, 'cropped')))
print('finished with {} annotations and {} images'.format(n, m))
