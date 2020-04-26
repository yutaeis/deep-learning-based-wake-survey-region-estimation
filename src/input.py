#model parameters
n_epoch = 200
batchsize = 8
lr = 1e-4

# input physical quantities
rho = 1. #density
u = 0.7 #velocity
p = 1./1.4 # pressure

#wing parameters
chord = 1. #chord length
sw = chord #wing area
max_dp_elem = 0.5 #Max entropy-drag element
min_dp_elem = 0 #Min entropy-drag element
len_pixel = 0.00510204081632653 #1 pixel length