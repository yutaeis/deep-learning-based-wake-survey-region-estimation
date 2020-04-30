import glob
import numpy as np
import pylab as plt

import chainer
import chainer.functions as F
from chainer import serializers

import input 
import module
import models

#-------------------------------------------
# make dataset
#-------------------------------------------

dataset = []

for path in glob.glob('../data/*/*.jpg'):
    airfoil, aoa, _, entropy = path.split('/')[-1].split('_')
    aoa = float(aoa.replace('aoa', ''))
    entropy = float(entropy.replace('.jpg', ''))

    data = {
        'path': path,
        'airfoil': airfoil,
        'aoa': aoa,
        'entropy': entropy
    }

    dataset.append(data)

# split train/test
train = []
test = []

for data in dataset:
    if data['airfoil'] == 'naca0012'or data['airfoil'] =='naca2413' or data['airfoil'] =='naca4414':
        test.append(data)
    else:
        train.append(data)

train = np.array(train)
test = np.array(test)

#-------------------------------------------
# model
#-------------------------------------------

model = models.Model()

optimizer = chainer.optimizers.Adam(alpha=input.lr)
optimizer.setup(model)

model.to_gpu()


logloss = []

for e in range(input.n_epoch):
    sum_loss = 0.
    perm = np.random.permutation(len(train))
    for i in range(0, len(train), input.batchsize):
        batch = train[perm[i:i+input.batchsize]]
        xs = []
        ts = []
        for b in batch:
            x, t = module.load(b)
            xs.append(x)
            ts.append(t)
        xs = model.xp.array(xs, dtype=np.float32)
        ts = model.xp.array(ts, dtype=np.float32)

        ys = model(xs)
        loss = F.mean_squared_error(ys, ts)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += loss.data

    print(e,sum_loss)
    logloss.append(sum_loss)

plt.plot(logloss)

SAVE_MODEL = 'mynet'
serializers.save_npz(SAVE_MODEL+'.npz', model)

