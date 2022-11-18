#import libraries

import os
import PIL
import random
import torchvision.transforms as T

# import configuration infos
from config import all_img_extn, factorX, HR_image_path, LR_image_path

def read_all_images(path, image_extentions=all_img_extn, HR_LR_resolutionfactor = factorX):
    '''
    INPUT
    path: path to the folder
    image_extention: all permissble extentions belonging to image class ex: jpeg, jpg etc
    HR_LR_resolutionfactor: downsampled resolution seeked, to ensure HR image can be resized by the particular factor, default: 2

    OUTPUT:
    all_image_files: list of all files that are image & their size are a factor of HR_LR_resolutionfactor
    '''
    # read all files
    all_files = os.listdir(path)
    # check if these files are images
    all_image_files = [os.path.join(path, x) for x in all_files if os.path.splitext(x)[1] in image_extentions]
    # check if image_files can be resized by the particular factor
    eligible_image_files = [x for x in all_image_files if ((PIL.Image.open(x).size[0]%HR_LR_resolutionfactor==0) and (PIL.Image.open(x).size[1]%HR_LR_resolutionfactor==0))]

    return eligible_image_files


def apply_random_transformation(image_path, HR_LR_resolutionfactor=factorX):
    '''
    INPUT:
    image_path: path of the image file

    OUTPUT:
    img_tr: transformed image post Gaussian Blur + Interpolation
    '''

    _interpolation_dict = {
        1: T.InterpolationMode.NEAREST,
        2: T.InterpolationMode.BILINEAR,
        3: T.InterpolationMode.BICUBIC,
        4: T.InterpolationMode.BOX,
        5: T.InterpolationMode.HAMMING,
        6: T.InterpolationMode.LANCZOS
    }
    # read the image
    base_img = PIL.Image.open(image_path)
    # get input image width & height
    w, h = base_img.size

    # blur the image
    blurred_img = T.GaussianBlur(kernel_size=(3, 3))(base_img)
    # LR the image
    random_int = random.randint(1, len(_interpolation_dict))
    lr_img = T.Resize((h // HR_LR_resolutionfactor, w // HR_LR_resolutionfactor),
                      interpolation=_interpolation_dict[random_int])(blurred_img)

    return lr_img


if __name__ == "__main__":
    # Get the filepaths of all eligible image files as a list
    all_HRimage_path = read_all_images(path= HR_image_path)

    #Iterate over all HR images and generate LR images
    for img_path in all_HRimage_path:
        # Get the corresponding LR image from HR image
        _lr_img = apply_random_transformation(image_path=img_path)
        # Save the image
        _lr_img.save(os.path.join(LR_image_path, os.path.basename(img_path)))
