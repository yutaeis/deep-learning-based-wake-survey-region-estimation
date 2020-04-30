
import chainer
import chainer.functions as F
import chainer.links as L


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