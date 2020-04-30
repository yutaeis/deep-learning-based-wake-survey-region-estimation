from PIL import Image
import numpy as np

import input


#-------------------------------------------
# dataset
#-------------------------------------------

# replace contor colors to Cd elements
def replace_cdelem(image, max_dp_elem, min_dp_elem):
    image = image.convert('L')
    image_array = (np.array(image, dtype=np.float32))

    # [0, 255] => [min_dp_elem, max_dp_elem]
    a = (max_dp_elem - min_dp_elem) / (255 - 0)
    image_array = a * (image_array - min_dp_elem) + min_dp_elem
    return image_array

# input wake survey region((x0,y0),(x1,y1)) and calculate drag
def dp_calc(image_array, p, rho, u):
    cdff = image_array.sum(axis=0) * p * input.len_pixel / (0.5 * rho * u**2)
    return cdff

#load data for learning
def load(data):
    image = Image.open(data['path'])

    cfd = replace_cdelem(image, input.max_dp_elem, input.min_dp_elem)
    cdff = dp_calc(cfd, input.p, input.rho, input.u)

    image = image.crop((0, 0, 1872, 980))
    image = image.resize((1872//4, 980//4), resample=Image.BILINEAR)
    image = np.array(image.convert('L'), dtype=np.float32)
    image = image[None]

    return image, (cdff / data['entropy'])[::4]
