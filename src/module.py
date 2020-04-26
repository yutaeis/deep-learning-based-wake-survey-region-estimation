from PIL import Image
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

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

#-------------------------------------------
# define model
#-------------------------------------------

class Model(chainer.Chain):

    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(1, 32, ksize=3, pad=1, stride=1)
            self.conv1 = L.Convolution2D(32, 64, ksize=3, pad=1, stride=1)
            self.conv2 = L.Convolution2D(64, 128, ksize=3, pad=1, stride=1)
            self.conv3 = L.Convolution2D(128, 256, ksize=3, pad=1, stride=1)
            self.conv4 = L.Convolution2D(256, 512, ksize=3, pad=1, stride=1)
            self.conv5 = L.Convolution2D(512, 512, ksize=3, pad=1, stride=1)
            self.conv6 = L.Convolution2D(512, 1, ksize=3, pad=1, stride=1)

    def __call__(self, x):
        h = F.relu(self.conv0(x))  # 245
        h = F.average_pooling_2d(h, ksize=[2, 1], stride=[2, 1], pad=0)  # 123
        h = F.relu(self.conv1(h))  # 123
        h = F.average_pooling_2d(h, ksize=[2, 1], stride=[2, 1], pad=0)  # 62
        h = F.relu(self.conv2(h))  # 62
        h = F.average_pooling_2d(h, ksize=[2, 1], stride=[2, 1], pad=0)  # 31
        h = F.relu(self.conv3(h))  # 31
        h = F.average_pooling_2d(h, ksize=[2, 1], stride=[2, 1], pad=0)  # 16
        h = F.relu(self.conv4(h))  # 16
        h = F.average_pooling_2d(h, ksize=[2, 1], stride=[2, 1], pad=0)  # 8
        h = F.relu(self.conv5(h))  # 8
        h = F.average_pooling_2d(h, ksize=[2, 1], stride=[2, 1], pad=0)  # 4
        h = F.softplus(self.conv6(h))  # 4
        h = F.average_pooling_2d(h, ksize=[3, 1], stride=[3, 1], pad=0)  # 2
        return h[:, 0, 0]
