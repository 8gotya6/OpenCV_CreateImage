import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import datetime
from Random_Shape import create_particle, inverse_img


def create_defect_sample(img_width, img_height, process_width, process_pos_range, is_save=False, save_dir='sample_image'):
    all_imgs = {}
    img_num = 0
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for process_pos in process_pos_range:
        # create a new image
        img = np.random.randint(150, 220, size=(img_height, img_width, 4), dtype=np.uint8)
        defect_qty = 0
        start_col = img_width//2 + process_pos
        # random create defect
        for side in [int(start_col - process_width//2), int(start_col + process_width//2)]:
            random_qty = np.random.randint(0, 2, size=1)[0]
            defect_pos = np.random.randint(0, 300, size=random_qty)
            defect_size = np.random.randint(3, 15, size=random_qty)
            random_color = np.random.randint(30, 50, size=(random_qty, 3))
            for pos, size, color in zip(defect_pos, defect_size, random_color):
                color = (int(color[0]), int(color[1]), int(color[2]))
                cv2.circle(img, (side, pos), size, color=(color[0], color[1], color[2]), thickness=-1)
                defect_qty += 1

        # normal process
        start_point = (start_col, 0)
        end_point = (start_col, img_height)
        img = cv2.line(img, start_point, end_point, (0, 0, 0), thickness=process_width)

        # add random shape defect
        new_defect_size = np.random.randint(30, 100, size=1)[0]
        new_defect = inverse_img(create_particle(new_defect_size))
        shift_pos = [
            np.random.randint(0, img_width - new_defect_size, size=1)[0], 
            np.random.randint(0, img_height - new_defect_size, size=1)[0]
        ]
        img = add_new_defect(img, new_defect, shift_pos)

        if defect_qty == 0:
            label = 'good'
        else:
            label = 'bad'

        img_name = str(img_num).zfill(4)
        all_imgs[img_name] = {
            'label': label,
            'img': img
        }
        
        if is_save:
            check_path(save_dir)
            fullpath = os.path.join(save_dir, f'{current_time}_{all_imgs[img_name]["label"]}_{img_name}.jpg')
            cv2.imwrite(fullpath, all_imgs[img_name]['img'])

        img_num += 1

    return  all_imgs


def check_path(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def add_new_defect(old_img, new_img, shift_xy:list):
    old_height, old_width, _ = old_img.shape
    new_height, new_width, _ = new_img.shape
    shift_x, shift_y = shift_xy

    if not old_width > new_width + shift_x or not old_height > new_height + shift_y:
        raise AttributeError(f'new_img is bigger than old_img')

    for w in range(old_width):
        for h in range(old_height):
            if w >= shift_x and w < shift_x + new_width and h > shift_y and h < shift_y + new_height:
                red, green, blue, alpha = new_img[w - shift_x, h - shift_y]
                if (red < 255).all() and (green < 255).all() and (blue < 255).all():
                    old_img[h, w] = [red, green, blue, alpha]
    
    return old_img



if __name__ == '__main__':
    #process param
    sampling_size = 10
    img_width = 400
    img_height = 300
    process_width = 10
    process_pos_range = np.random.randint(-10, 10, size=sampling_size)

    create_defect_sample(img_width, img_height, process_width, process_pos_range, is_save=True)