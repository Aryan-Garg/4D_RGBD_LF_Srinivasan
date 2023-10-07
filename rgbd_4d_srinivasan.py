#!/usr/bin/env python3

import os
import sys

import warnings
warnings.filterwarnings("ignore")

import numpy as np
# import cv2

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from PIL import Image

import scipy as sp
from scipy import io
from scipy import interpolate
from scipy import ndimage

import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import ImageFolder, Dataset, DataLoader


import time

#parameters
# -------------------------------------------------------------------
lfsize = [192, 192, 7, 7] # dimensions of Lytro light fields
batchsize = 1 # modify based on user's GPU memory
patchsize = [192, 192] # spatial dimensions of training light fields
disp_mult = 4.0 # max disparity between adjacent veiws
num_crops = 4 # number of random spatial crops per light field for each input queue thread to push
learning_rate = 0.001
train_iters = 120000
# -------------------------------------------------------------------


# functions for CNN layers
# -------------------------------------------------------------------
def weight_variable(w_shape):
    return tf.get_variable('weights', w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


def torch_weight_variable(w_shape):
    return torch.nn.init.xavier_uniform_(torch.empty(w_shape))


def bias_variable(b_shape, init_bias=0.0):
    return tf.get_variable('bias', b_shape, initializer=tf.constant_initializer(init_bias))


def torch_bias_variable(b_shape, init_bias=0.0):
    return torch.nn.init.constant_(torch.empty(b_shape), init_bias)


def cnn_layer(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    with tf.variable_scope(layer_name):
        W = weight_variable(w_shape)
        pad_amt_0 = rate * (w_shape[0] - 1)//2
        pad_amt_1 = rate * (w_shape[1] - 1)//2
        input_tensor = tf.pad(input_tensor, [[0,0],[pad_amt_0,pad_amt_0],[pad_amt_1,pad_amt_1],[0,0]], mode='SYMMETRIC')
        h = tf.nn.convolution(input_tensor, W, strides=[ds, ds], padding='VALID', dilation_rate=[rate, rate], name=layer_name + '_conv')
        h = tf.contrib.layers.instance_norm(h + bias_variable(b_shape))
        h = tf.nn.leaky_relu(h)
    
        return h

def torch_cnn_layer(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    W = torch_weight_variable(w_shape)
    pad_amt_0 = rate * (w_shape[0] - 1)//2
    pad_amt_1 = rate * (w_shape[1] - 1)//2
    input_tensor = F.pad(input_tensor, (pad_amt_1, pad_amt_1, pad_amt_0, pad_amt_0), mode='reflect')
    h = F.conv2d(input_tensor, W, stride=ds, padding='valid', dilation=rate)
    h = F.instance_norm(h + torch_bias_variable(b_shape))
    h = F.leaky_relu(h)

    return h


def cnn_layer_plain(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    with tf.variable_scope(layer_name):
        W = weight_variable(w_shape)
        pad_amt_0 = rate * (w_shape[0] - 1)//2
        pad_amt_1 = rate * (w_shape[1] - 1)//2
        input_tensor = tf.pad(input_tensor, [[0,0],[pad_amt_0,pad_amt_0],[pad_amt_1,pad_amt_1],[0,0]], mode='SYMMETRIC')
        h = tf.nn.convolution(input_tensor, W, strides=[ds, ds], padding='VALID', dilation_rate=[rate, rate], name=layer_name + '_conv')
        h = h + bias_variable(b_shape)
        return h
    

def torch_cnn_layer_plain(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    W = torch_weight_variable(w_shape)
    pad_amt_0 = rate * (w_shape[0] - 1)//2
    pad_amt_1 = rate * (w_shape[1] - 1)//2
    input_tensor = F.pad(input_tensor, (pad_amt_1, pad_amt_1, pad_amt_0, pad_amt_0), mode='reflect')
    h = F.conv2d(input_tensor, W, stride=ds, padding='valid', dilation=rate)
    h = h + torch_bias_variable(b_shape)

    return h


def cnn_layer_3D(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    with tf.variable_scope(layer_name):
        W = weight_variable(w_shape)
        pad_amt_0 = rate * (w_shape[0] - 1)//2
        pad_amt_1 = rate * (w_shape[1] - 1)//2
        pad_amt_2 = rate * (w_shape[2] - 1)//2
        input_tensor = tf.pad(input_tensor, [[0,0],[pad_amt_0,pad_amt_0],[pad_amt_1,pad_amt_1],[pad_amt_2,pad_amt_2],[0,0]], mode='SYMMETRIC')
        h = tf.nn.convolution(input_tensor, W, strides=[ds, ds, ds], padding='VALID', dilation_rate=[rate, rate, rate], name=layer_name + '_conv')
        h = tf.contrib.layers.instance_norm(h + bias_variable(b_shape))
        h = tf.nn.leaky_relu(h)
        return h


def torch_cnn_layer_3D(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    W = torch_weight_variable(w_shape)
    pad_amt_0 = rate * (w_shape[0] - 1)//2
    pad_amt_1 = rate * (w_shape[1] - 1)//2
    pad_amt_2 = rate * (w_shape[2] - 1)//2
    input_tensor = F.pad(input_tensor, (pad_amt_2, pad_amt_2, pad_amt_1, pad_amt_1, pad_amt_0, pad_amt_0), mode='reflect')
    h = F.conv3d(input_tensor, W, stride=ds, padding='valid', dilation=rate)
    h = F.instance_norm(h + torch_bias_variable(b_shape))
    h = F.leaky_relu(h)

    return h


def cnn_layer_3D_plain(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    with tf.variable_scope(layer_name):
        W = weight_variable(w_shape)
        pad_amt_0 = rate * (w_shape[0] - 1)//2
        pad_amt_1 = rate * (w_shape[1] - 1)//2
        pad_amt_2 = rate * (w_shape[2] - 1)//2
        input_tensor = tf.pad(input_tensor, [[0,0],[pad_amt_0,pad_amt_0],[pad_amt_1,pad_amt_1],[pad_amt_2,pad_amt_2],[0,0]], mode='SYMMETRIC')
        h = tf.nn.convolution(input_tensor, W, strides=[ds, ds, ds], padding='VALID', dilation_rate=[rate, rate, rate], name=layer_name + '_conv')
        h = h + bias_variable(b_shape)
        return h
    

def torch_cnn_layer_3D_plain(input_tensor, w_shape, b_shape, layer_name, rate=1, ds=1):
    W = torch_weight_variable(w_shape)
    pad_amt_0 = rate * (w_shape[0] - 1)//2
    pad_amt_1 = rate * (w_shape[1] - 1)//2
    pad_amt_2 = rate * (w_shape[2] - 1)//2
    input_tensor = F.pad(input_tensor, (pad_amt_2, pad_amt_2, pad_amt_1, pad_amt_1, pad_amt_0, pad_amt_0), mode='reflect')
    h = F.conv3d(input_tensor, W, stride=ds, padding='valid', dilation=rate)
    h = h + torch_bias_variable(b_shape)

    return h
# -------------------------------------------------------------------

#network to predict ray depths from input image
# -------------------------------------------------------------------
def depth_network(x, lfsize, disp_mult, name):
    with tf.variable_scope(name):

        b_sz = tf.shape(x)[0]
        y_sz = tf.shape(x)[1]
        x_sz = tf.shape(x)[2]
        v_sz = lfsize[2]
        u_sz = lfsize[3]

        c1 = cnn_layer(x, [3, 3, 3, 16], [16], 'c1')
        c2 = cnn_layer(c1, [3, 3, 16, 64], [64], 'c2')
        c3 = cnn_layer(c2, [3, 3, 64, 128], [128], 'c3')
        c4 = cnn_layer(c3, [3, 3, 128, 128], [128], 'c4', rate=2)
        c5 = cnn_layer(c4, [3, 3, 128, 128], [128], 'c5', rate=4)
        c6 = cnn_layer(c5, [3, 3, 128, 128], [128], 'c6', rate=8)
        c7 = cnn_layer(c6, [3, 3, 128, 128], [128], 'c7', rate=16)
        c8 = cnn_layer(c7, [3, 3, 128, 128], [128], 'c8')
        c9 = cnn_layer(c8, [3, 3, 128, lfsize[2]*lfsize[3]], [lfsize[2]*lfsize[3]], 'c9')
        c10 = disp_mult*tf.tanh(cnn_layer_plain(c9, [3, 3, lfsize[2]*lfsize[3], lfsize[2]*lfsize[3]], \
                                                [lfsize[2]*lfsize[3]], 'c10'))

        return tf.reshape(c10, [b_sz, y_sz, x_sz, v_sz, u_sz])


def torch_depth_network(x, lfsize, disp_mult, name):
    b_sz = x.shape[0]
    y_sz = x.shape[1]
    x_sz = x.shape[2]
    v_sz = lfsize[2]
    u_sz = lfsize[3]

    c1 = torch_cnn_layer(x, [3, 3, 3, 16], [16], 'c1')
    c2 = torch_cnn_layer(c1, [3, 3, 16, 64], [64], 'c2')
    c3 = torch_cnn_layer(c2, [3, 3, 64, 128], [128], 'c3')
    c4 = torch_cnn_layer(c3, [3, 3, 128, 128], [128], 'c4', rate=2)
    c5 = torch_cnn_layer(c4, [3, 3, 128, 128], [128], 'c5', rate=4)
    c6 = torch_cnn_layer(c5, [3, 3, 128, 128], [128], 'c6', rate=8)
    c7 = torch_cnn_layer(c6, [3, 3, 128, 128], [128], 'c7', rate=16)
    c8 = torch_cnn_layer(c7, [3, 3, 128, 128], [128], 'c8')
    c9 = torch_cnn_layer(c8, [3, 3, 128, lfsize[2]*lfsize[3]], [lfsize[2]*lfsize[3]], 'c9')
    c10 = disp_mult*torch.tanh(torch_cnn_layer_plain(c9, [3, 3, lfsize[2]*lfsize[3], lfsize[2]*lfsize[3]], \
                                            [lfsize[2]*lfsize[3]], 'c10'))
    
    return c10.view(b_sz, y_sz, x_sz, v_sz, u_sz)


class DepthNetwork(nn.Module):
    def __init__(self, lfsize, disp_mult, name):
        super().__init__()
        self.lfsize = lfsize
        self.disp_mult = disp_mult
        self.name = name

        self.v_sz = self.lfsize[2]
        self.u_sz = self.lfsize[3]


    def forward(self, x):
        b_sz = x.shape[0]
        y_sz = x.shape[1]
        x_sz = x.shape[2]

        c1 = torch_cnn_layer(x, [3, 3, 3, 16], [16], 'c1')
        c2 = torch_cnn_layer(c1, [3, 3, 16, 64], [64], 'c2')
        c3 = torch_cnn_layer(c2, [3, 3, 64, 128], [128], 'c3')
        c4 = torch_cnn_layer(c3, [3, 3, 128, 128], [128], 'c4', rate=2)
        c5 = torch_cnn_layer(c4, [3, 3, 128, 128], [128], 'c5', rate=4)
        c6 = torch_cnn_layer(c5, [3, 3, 128, 128], [128], 'c6', rate=8)
        c7 = torch_cnn_layer(c6, [3, 3, 128, 128], [128], 'c7', rate=16)
        c8 = torch_cnn_layer(c7, [3, 3, 128, 128], [128], 'c8')
        c9 = torch_cnn_layer(c8, [3, 3, 128, lfsize[2]*lfsize[3]], [lfsize[2]*lfsize[3]], 'c9')
        c10 = disp_mult*torch.tanh(torch_cnn_layer_plain(c9, [3, 3, lfsize[2]*lfsize[3], lfsize[2]*lfsize[3]], \
                                                [lfsize[2]*lfsize[3]], 'c10'))

        return c10.view(b_sz, y_sz, x_sz, self.v_sz, self.u_sz)



# -------------------------------------------------------------------

#network for refining Lambertian light field (predict occluded rays and non-Lambertian effects)
# -------------------------------------------------------------------
def occlusions_network(x, shear, lfsize, name):
    with tf.variable_scope(name):

        b_sz = tf.shape(x)[0]
        y_sz = tf.shape(x)[1]
        x_sz = tf.shape(x)[2]
        v_sz = lfsize[2]
        u_sz = lfsize[3]

        x = tf.transpose(tf.reshape(tf.transpose(x, perm=[0, 5, 1, 2, 3, 4]), \
                                    [b_sz, 4, y_sz, x_sz, u_sz*v_sz]), perm=[0, 4, 2, 3, 1])

        c1 = cnn_layer_3D(x, [3, 3, 3, 4, 8], [8], 'c1')
        c2 = cnn_layer_3D(c1, [3, 3, 3, 8, 8], [8], 'c2')
        c3 = cnn_layer_3D(c2, [3, 3, 3, 8, 8], [8], 'c3')
        c4 = cnn_layer_3D(c3, [3, 3, 3, 8, 8], [8], 'c4')
        c5 = tf.tanh(cnn_layer_3D_plain(c4, [3, 3, 3, 8, 3], [3], 'c5'))

        output = tf.transpose(tf.reshape(tf.transpose(c5, perm=[0, 4, 2, 3, 1]), \
                                         [b_sz, 3, y_sz, x_sz, v_sz, u_sz]), perm=[0, 2, 3, 4, 5, 1]) + shear

        return output
    

def torch_occlusions_network(x, shear, lfsize, name):
    b_sz = x.shape[0]
    y_sz = x.shape[1]
    x_sz = x.shape[2]
    v_sz = lfsize[2]
    u_sz = lfsize[3]

    x = x.permute(0, 5, 1, 2, 3, 4).reshape(b_sz, 4, y_sz, x_sz, u_sz*v_sz).permute(0, 4, 2, 3, 1)
    
    c1 = torch_cnn_layer_3D(x, [3, 3, 3, 4, 8], [8], 'c1')
    c2 = torch_cnn_layer_3D(c1, [3, 3, 3, 8, 8], [8], 'c2')
    c3 = torch_cnn_layer_3D(c2, [3, 3, 3, 8, 8], [8], 'c3')
    c4 = torch_cnn_layer_3D(c3, [3, 3, 3, 8, 8], [8], 'c4')
    c5 = torch.tanh(torch_cnn_layer_3D_plain(c4, [3, 3, 3, 8, 3], [3], 'c5'))

    output = c5.permute(0, 4, 2, 3, 1).reshape(b_sz, 3, y_sz, x_sz, v_sz, u_sz).permute(0, 2, 3, 4, 5, 1) + shear

    return output
# -------------------------------------------------------------------


#render light field from input image and ray depths
# -------------------------------------------------------------------
def depth_rendering(central, ray_depths, lfsize):
    with tf.variable_scope('depth_rendering') as scope:
        b_sz = tf.shape(central)[0]
        y_sz = tf.shape(central)[1]
        x_sz = tf.shape(central)[2]
        u_sz = lfsize[2]
        v_sz = lfsize[3]

        central = tf.expand_dims(tf.expand_dims(central, 3), 4)

        #create and reparameterize light field grid
        b_vals = tf.to_float(tf.range(b_sz))
        v_vals = tf.to_float(tf.range(v_sz)) - tf.to_float(v_sz)/2.0
        u_vals = tf.to_float(tf.range(u_sz)) - tf.to_float(u_sz)/2.0
        y_vals = tf.to_float(tf.range(y_sz))
        x_vals = tf.to_float(tf.range(x_sz))

        b, y, x, v, u = tf.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals, indexing='ij')

        #warp coordinates by ray depths
        y_t = y + v * ray_depths
        x_t = x + u * ray_depths

        v_r = tf.zeros_like(b)
        u_r = tf.zeros_like(b)

        #indices for linear interpolation
        b_1 = tf.to_int32(b)
        y_1 = tf.to_int32(tf.floor(y_t))
        y_2 = y_1 + 1
        x_1 = tf.to_int32(tf.floor(x_t))
        x_2 = x_1 + 1
        v_1 = tf.to_int32(v_r)
        u_1 = tf.to_int32(u_r)

        y_1 = tf.clip_by_value(y_1, 0, y_sz-1)
        y_2 = tf.clip_by_value(y_2, 0, y_sz-1)
        x_1 = tf.clip_by_value(x_1, 0, x_sz-1)
        x_2 = tf.clip_by_value(x_2, 0, x_sz-1)

        #assemble interpolation indices
        interp_pts_1 = tf.stack([b_1, y_1, x_1, v_1, u_1], -1)
        interp_pts_2 = tf.stack([b_1, y_2, x_1, v_1, u_1], -1)
        interp_pts_3 = tf.stack([b_1, y_1, x_2, v_1, u_1], -1)
        interp_pts_4 = tf.stack([b_1, y_2, x_2, v_1, u_1], -1)

        #gather light fields to be interpolated
        lf_1 = tf.gather_nd(central, interp_pts_1)
        lf_2 = tf.gather_nd(central, interp_pts_2)
        lf_3 = tf.gather_nd(central, interp_pts_3)
        lf_4 = tf.gather_nd(central, interp_pts_4)

        #calculate interpolation weights
        y_1_f = tf.to_float(y_1)
        x_1_f = tf.to_float(x_1)
        d_y_1 = 1.0 - (y_t - y_1_f)
        d_y_2 = 1.0 - d_y_1
        d_x_1 = 1.0 - (x_t - x_1_f)
        d_x_2 = 1.0 - d_x_1

        w1 = d_y_1 * d_x_1
        w2 = d_y_2 * d_x_1
        w3 = d_y_1 * d_x_2
        w4 = d_y_2 * d_x_2

        lf = tf.add_n([w1*lf_1, w2*lf_2, w3*lf_3, w4*lf_4])

    return lf


def torch_depth_rendering(central, ray_depths, lfsize):
    b_sz = central.shape[0]
    y_sz = central.shape[1]
    x_sz = central.shape[2]
    u_sz = lfsize[2]
    v_sz = lfsize[3]

    central = central.unsqueeze(3).unsqueeze(4)

    #create and reparameterize light field grid
    b_vals = torch.arange(b_sz).float()
    v_vals = torch.arange(v_sz).float() - v_sz/2.0
    u_vals = torch.arange(u_sz).float() - u_sz/2.0
    y_vals = torch.arange(y_sz).float()
    x_vals = torch.arange(x_sz).float()

    b, y, x, v, u = torch.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals)

    #warp coordinates by ray depths
    y_t = y + v * ray_depths
    x_t = x + u * ray_depths

    v_r = torch.zeros_like(b)
    u_r = torch.zeros_like(b)

    #indices for linear interpolation
    b_1 = b.long()
    y_1 = y_t.long()
    y_2 = y_1 + 1
    x_1 = x_t.long()
    x_2 = x_1 + 1
    v_1 = v_r.long()
    u_1 = u_r.long()

    y_1 = torch.clamp(y_1, 0, y_sz-1)
    y_2 = torch.clamp(y_2, 0, y_sz-1)
    x_1 = torch.clamp(x_1, 0, x_sz-1)
    x_2 = torch.clamp(x_2, 0, x_sz-1)

    #assemble interpolation indices
    interp_pts_1 = torch.stack([b_1, y_1, x_1, v_1, u_1], -1)
    interp_pts_2 = torch.stack([b_1, y_2, x_1, v_1, u_1], -1)
    interp_pts_3 = torch.stack([b_1, y_1, x_2, v_1, u_1], -1)
    interp_pts_4 = torch.stack([b_1, y_2, x_2, v_1, u_1], -1)
    
    #gather light fields to be interpolated
    lf_1 = torch.gather(central, 1, interp_pts_1)
    lf_2 = torch.gather(central, 1, interp_pts_2)
    lf_3 = torch.gather(central, 1, interp_pts_3)
    lf_4 = torch.gather(central, 1, interp_pts_4)

    #calculate interpolation weights
    y_1_f = y_1.float()
    x_1_f = x_1.float()
    d_y_1 = 1.0 - (y_t - y_1_f)
    d_y_2 = 1.0 - d_y_1
    d_x_1 = 1.0 - (x_t - x_1_f)
    d_x_2 = 1.0 - d_x_1

    w1 = d_y_1 * d_x_1
    w2 = d_y_2 * d_x_1
    w3 = d_y_1 * d_x_2
    w4 = d_y_2 * d_x_2

    lf = w1*lf_1 + w2*lf_2 + w3*lf_3 + w4*lf_4

    return lf.squeeze(1)

# -------------------------------------------------------------------

#full forward model
# -------------------------------------------------------------------
def forward_model(x, lfsize, disp_mult):
    with tf.variable_scope('forward_model') as scope:
        #predict ray depths from input image
        ray_depths = depth_network(x, lfsize, disp_mult, 'ray_depths')
        #shear input image by predicted ray depths to render Lambertian light field
        lf_shear_r = depth_rendering(x[:, :, :, 0], ray_depths, lfsize)
        lf_shear_g = depth_rendering(x[:, :, :, 1], ray_depths, lfsize)
        lf_shear_b = depth_rendering(x[:, :, :, 2], ray_depths, lfsize)
        lf_shear = tf.stack([lf_shear_r, lf_shear_g, lf_shear_b], axis=5)
        #occlusion/non-Lambertian prediction network
        shear_and_depth = tf.stack([lf_shear_r, lf_shear_g, lf_shear_b, tf.stop_gradient(ray_depths)], axis=5)
        y = occlusions_network(shear_and_depth, lf_shear, lfsize, 'occlusions')
        return ray_depths, lf_shear, y
    

def torch_forward_model(x, lfsize, disp_mult):
    ray_depths = torch_depth_network(x, lfsize, disp_mult, 'ray_depths')
    lf_shear_r = torch_depth_rendering(x[:, :, :, 0], ray_depths, lfsize)
    lf_shear_g = torch_depth_rendering(x[:, :, :, 1], ray_depths, lfsize)
    lf_shear_b = torch_depth_rendering(x[:, :, :, 2], ray_depths, lfsize)
    lf_shear = torch.stack([lf_shear_r, lf_shear_g, lf_shear_b], dim=5)

    shear_and_depth = torch.stack([lf_shear_r, lf_shear_g, lf_shear_b, ray_depths.detach()], dim=5)
    y = torch_occlusions_network(shear_and_depth, lf_shear, lfsize, 'occlusions')
    return ray_depths, lf_shear, y


class SrinivasanNet(nn.Module):
    def __init__(self, lfsize, disp_mult, ray_depths):
        super().__init__()
        self.depth_network = DepthNetwork(lfsize, disp_mult, 'depth_net')

    def forward(self, x):
        ray_depths = self.depth_network(x)
        lf_shear_r = torch_depth_rendering(x[:, :, :, 0], ray_depths, lfsize)
        
# -------------------------------------------------------------------

#resample ray depths for depth consistency regularization
# -------------------------------------------------------------------
def transform_ray_depths(ray_depths, u_step, v_step, lfsize):
    with tf.variable_scope('transform_ray_depths') as scope:
        b_sz = tf.shape(ray_depths)[0]
        y_sz = tf.shape(ray_depths)[1]
        x_sz = tf.shape(ray_depths)[2]
        u_sz = lfsize[2]
        v_sz = lfsize[3]

        #create and reparameterize light field grid
        b_vals = tf.to_float(tf.range(b_sz))
        v_vals = tf.to_float(tf.range(v_sz)) - tf.to_float(v_sz)/2.0
        u_vals = tf.to_float(tf.range(u_sz)) - tf.to_float(u_sz)/2.0
        y_vals = tf.to_float(tf.range(y_sz))
        x_vals = tf.to_float(tf.range(x_sz))

        b, y, x, v, u = tf.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals, indexing='ij')

        #warp coordinates by ray depths
        y_t = y + v_step * ray_depths
        x_t = x + u_step * ray_depths

        v_t = v - v_step + tf.to_float(v_sz)/2.0
        u_t = u - u_step + tf.to_float(u_sz)/2.0

        #indices for linear interpolation
        b_1 = tf.to_int32(b)
        y_1 = tf.to_int32(tf.floor(y_t))
        y_2 = y_1 + 1
        x_1 = tf.to_int32(tf.floor(x_t))
        x_2 = x_1 + 1
        v_1 = tf.to_int32(v_t)
        u_1 = tf.to_int32(u_t)

        y_1 = tf.clip_by_value(y_1, 0, y_sz-1)
        y_2 = tf.clip_by_value(y_2, 0, y_sz-1)
        x_1 = tf.clip_by_value(x_1, 0, x_sz-1)
        x_2 = tf.clip_by_value(x_2, 0, x_sz-1)
        v_1 = tf.clip_by_value(v_1, 0, v_sz-1)
        u_1 = tf.clip_by_value(u_1, 0, u_sz-1)

        #assemble interpolation indices
        interp_pts_1 = tf.stack([b_1, y_1, x_1, v_1, u_1], -1)
        interp_pts_2 = tf.stack([b_1, y_2, x_1, v_1, u_1], -1)
        interp_pts_3 = tf.stack([b_1, y_1, x_2, v_1, u_1], -1)
        interp_pts_4 = tf.stack([b_1, y_2, x_2, v_1, u_1], -1)

        #gather light fields to be interpolated
        lf_1 = tf.gather_nd(ray_depths, interp_pts_1)
        lf_2 = tf.gather_nd(ray_depths, interp_pts_2)
        lf_3 = tf.gather_nd(ray_depths, interp_pts_3)
        lf_4 = tf.gather_nd(ray_depths, interp_pts_4)

        #calculate interpolation weights
        y_1_f = tf.to_float(y_1)
        x_1_f = tf.to_float(x_1)
        d_y_1 = 1.0 - (y_t - y_1_f)
        d_y_2 = 1.0 - d_y_1
        d_x_1 = 1.0 - (x_t - x_1_f)
        d_x_2 = 1.0 - d_x_1

        w1 = d_y_1 * d_x_1
        w2 = d_y_2 * d_x_1
        w3 = d_y_1 * d_x_2
        w4 = d_y_2 * d_x_2

        lf = tf.add_n([w1*lf_1, w2*lf_2, w3*lf_3, w4*lf_4])
    return lf


def torch_transform_ray_depths(ray_depths, u_step, v_step, lfsize):
    b_sz = ray_depths.shape[0]
    y_sz = ray_depths.shape[1]
    x_sz = ray_depths.shape[2]
    u_sz = lfsize[2]
    v_sz = lfsize[3]

    #create and reparameterize light field grid
    b_vals = torch.arange(b_sz).float()
    v_vals = torch.arange(v_sz).float() - v_sz/2.0
    u_vals = torch.arange(u_sz).float() - u_sz/2.0
    y_vals = torch.arange(y_sz).float()
    x_vals = torch.arange(x_sz).float()

    b, y, x, v, u = torch.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals)

    #warp coordinates by ray depths
    y_t = y + v_step * ray_depths
    x_t = x + u_step * ray_depths

    v_t = v - v_step + v_sz/2.0
    u_t = u - u_step + u_sz/2.0

    #indices for linear interpolation
    b_1 = b.long()
    y_1 = y_t.long()
    y_2 = y_1 + 1
    x_1 = x_t.long()
    x_2 = x_1 + 1
    v_1 = v_t.long()
    u_1 = u_t.long()

    y_1 = torch.clamp(y_1, 0, y_sz-1)
    y_2 = torch.clamp(y_2, 0, y_sz-1)
    x_1 = torch.clamp(x_1, 0, x_sz-1)
    x_2 = torch.clamp(x_2, 0, x_sz-1)
    v_1 = torch.clamp(v_1, 0, v_sz-1)
    u_1 = torch.clamp(u_1, 0, u_sz-1)

    #assemble interpolation indices
    interp_pts_1 = torch.stack([b_1, y_1, x_1, v_1, u_1], -1)
    interp_pts_2 = torch.stack([b_1, y_2, x_1, v_1, u_1], -1)
    interp_pts_3 = torch.stack([b_1, y_1, x_2, v_1, u_1], -1)
    interp_pts_4 = torch.stack([b_1, y_2, x_2, v_1, u_1], -1)

    #gather light fields to be interpolated
    lf_1 = torch.gather(ray_depths, 1, interp_pts_1)
    lf_2 = torch.gather(ray_depths, 1, interp_pts_2)
    lf_3 = torch.gather(ray_depths, 1, interp_pts_3)
    lf_4 = torch.gather(ray_depths, 1, interp_pts_4)

    #calculate interpolation weights
    y_1_f = y_1.float()
    x_1_f = x_1.float()
    d_y_1 = 1.0 - (y_t - y_1_f)
    d_y_2 = 1.0 - d_y_1
    d_x_1 = 1.0 - (x_t - x_1_f)
    d_x_2 = 1.0 - d_x_1

    w1 = d_y_1 * d_x_1
    w2 = d_y_2 * d_x_1
    w3 = d_y_1 * d_x_2
    w4 = d_y_2 * d_x_2

    lf = w1*lf_1 + w2*lf_2 + w3*lf_3 + w4*lf_4

    return lf.squeeze(1)

# -------------------------------------------------------------------

#loss to encourage consistency of ray depths corresponding to same scene point
# -------------------------------------------------------------------
def depth_consistency_loss(x, lfsize):
    x_u = transform_ray_depths(x, 1.0, 0.0, lfsize)
    x_v = transform_ray_depths(x, 0.0, 1.0, lfsize)
    x_uv = transform_ray_depths(x, 1.0, 1.0, lfsize)
    d1 = (x[:,:,:,1:,1:]-x_u[:,:,:,1:,1:])
    d2 = (x[:,:,:,1:,1:]-x_v[:,:,:,1:,1:])
    d3 = (x[:,:,:,1:,1:]-x_uv[:,:,:,1:,1:])
    l1 = tf.reduce_mean(tf.abs(d1)+tf.abs(d2)+tf.abs(d3))
    return l1


def torch_depth_consistency_loss(x, lfsize):
    x_u = torch_transform_ray_depths(x, 1.0, 0.0, lfsize)
    x_v = torch_transform_ray_depths(x, 0.0, 1.0, lfsize)
    x_uv = torch_transform_ray_depths(x, 1.0, 1.0, lfsize)
    d1 = (x[:,1:,1:]-x_u[:,1:,1:])
    d2 = (x[:,1:,1:]-x_v[:,1:,1:])
    d3 = (x[:,1:,1:]-x_uv[:,1:,1:])
    l1 = torch.mean(torch.abs(d1)+torch.abs(d2)+torch.abs(d3))
    return l1

# -------------------------------------------------------------------

#spatial TV loss (l1 of spatial derivatives)
# -------------------------------------------------------------------
def image_derivs(x, nc):
    dy = tf.nn.depthwise_conv2d(x, tf.tile(tf.expand_dims(tf.expand_dims([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], 2), 3), [1, 1, nc, 1]), strides=[1, 1, 1, 1], padding='VALID')
    dx = tf.nn.depthwise_conv2d(x, tf.tile(tf.expand_dims(tf.expand_dims([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], 2), 3), [1, 1, nc, 1]), strides=[1, 1, 1, 1], padding='VALID')
    return dy, dx


def torch_image_derivs(x, nc):
    dy = torch.nn.functional.conv2d(x, torch.tile(torch.unsqueeze(torch.unsqueeze(torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], 2), 3), [1, 1, nc, 1]), stride=1, padding=0))
    dx = torch.nn.functional.conv2d(x, torch.tile(torch.unsqueeze(torch.unsqueeze(torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], 2), 3), [1, 1, nc, 1]), stride=1, padding=0))
    return dy, dx

                            
def tv_loss(x):
    b_sz = tf.shape(x)[0]
    y_sz = tf.shape(x)[1]
    x_sz = tf.shape(x)[2]
    u_sz = lfsize[2]
    v_sz = lfsize[3]
    temp = tf.reshape(x, [b_sz, y_sz, x_sz, u_sz*v_sz])
    dy, dx = image_derivs(temp, u_sz*v_sz)
    l1 = tf.reduce_mean(tf.abs(dy)+tf.abs(dx))
    return l1


def torch_tv_loss(x):
    b_sz = x.shape[0]
    y_sz = x.shape[1]
    x_sz = x.shape[2]
    u_sz = lfsize[2]
    v_sz = lfsize[3]
    temp = x.view(b_sz, y_sz, x_sz, u_sz*v_sz)
    dy, dx = torch_image_derivs(temp, u_sz*v_sz)
    l1 = torch.mean(torch.abs(dy)+torch.abs(dx))
    return l1
# -------------------------------------------------------------------

#normalize to between -1 and 1, given input between 0 and 1
# -------------------------------------------------------------------
def normalize_lf(lf):
    return 2.0*(lf-0.5)

# Haha no changes here!
def torch_normalize_lf(lf):
    return 2.0*(lf-0.5)
# -------------------------------------------------------------------

#input pipeline
# -------------------------------------------------------------------
def process_lf(lf, num_crops, lfsize, patchsize):
    gamma_val = tf.random_uniform(shape=[], minval=0.4, maxval=1.0) #random gamma for data augmentation (change at test time, I suggest 0.4-0.5)
    lf = normalize_lf(tf.image.adjust_gamma(tf.to_float(lf[:lfsize[0]*14, :lfsize[1]*14, :])/255.0, gamma=gamma_val))
    lf = tf.transpose(tf.reshape(lf, [lfsize[0], 14, lfsize[1], 14, 3]), [0, 2, 1, 3, 4])
    # print("LF_orig:", lf.shape)
    lf = lf[:, :, 4:11, 4:11, :]
    # print("LF:", lf.shape)
    aif = lf[:, :, 7//2, 7//2, :]
    # print("AIF:", aif.shape)
    aif_list = []
    lf_list = []
    for i in range(num_crops):
        r = tf.random_uniform(shape=[], minval=0, maxval=tf.shape(lf)[0]-patchsize[0], dtype=tf.int32)
        c = tf.random_uniform(shape=[], minval=0, maxval=tf.shape(lf)[1]-patchsize[1], dtype=tf.int32)
        aif_list.append(aif[r:r+patchsize[0], c:c+patchsize[1], :])
        lf_list.append(lf[r:r+patchsize[0], c:c+patchsize[1], :, :, :])
    return aif_list, lf_list


def torch_process_lf(lf, num_crops, lfsize, patchsize):
    gamma_val = torch.rand(1)*(1.0-0.4)+0.4
    lf = torch_normalize_lf(torch.pow(lf[:lfsize[0]*14, :lfsize[1]*14, :]/255.0, gamma_val))
    lf = lf.view(lfsize[0], 14, lfsize[1], 14, 3).permute(0, 2, 1, 3, 4)
    lf = lf[:, :, 4:11, 4:11, :]
    aif = lf[:, :, 7//2, 7//2, :]
    aif_list = []
    lf_list = []
    for i in range(num_crops):
        r = torch.randint(0, lf.shape[0]-patchsize[0], (1,))
        c = torch.randint(0, lf.shape[1]-patchsize[1], (1,))
        aif_list.append(aif[r:r+patchsize[0], c:c+patchsize[1], :])
        lf_list.append(lf[r:r+patchsize[0], c:c+patchsize[1], :, :, :])
    return aif_list, lf_list


def read_lf(filename_queue, num_crops, lfsize, patchsize):
    value = tf.read_file(filename_queue[0])
    lf = tf.image.decode_image(value, channels=3)
    aif_list, lf_list = process_lf(lf, num_crops, lfsize, patchsize)
    return aif_list, lf_list


def torch_read_lf(filename_queue, num_crops, lfsize, patchsize):
    lf = torch.from_numpy(np.array(Image.open(filename_queue[0]))).float()
    aif_list, lf_list = torch_process_lf(lf, num_crops, lfsize, patchsize)
    return aif_list, lf_list


def input_pipeline(filenames, lfsize, patchsize, batchsize, num_crops):
    filename_queue = tf.train.slice_input_producer([filenames], shuffle=True)
    example_list = [read_lf(filename_queue, num_crops, lfsize, patchsize) for _ in range(2)] #number of threads for populating queue
    capacity = 1
    aif_batch, lf_batch = tf.train.shuffle_batch_join(example_list, batch_size=batchsize, capacity=capacity, 
                                                      min_after_dequeue=0,
                                                      shapes=[[patchsize[0], patchsize[1], 3],
                                                              [patchsize[0], patchsize[1], lfsize[2], lfsize[3], 3]],
                                                        allow_smaller_final_batch=True)
    return aif_batch, lf_batch


def torch_input_pipeline(filenames, lfsize, patchsize, batchsize, num_crops):
    example_list = [torch_read_lf([filename], num_crops, lfsize, patchsize) for filename in filenames]
    aif_batch, lf_batch = torch.utils.data.DataLoader(example_list, batch_size=batchsize, shuffle=True)
    return aif_batch, lf_batch
# -------------------------------------------------------------------

train_path = '/data/prasan/datasets/LF_datasets/TAMULF/train' # path to training examples
test_path = '/data/prasan/datasets/LF_datasets/TAMULF/test/'

train_filenames = [os.path.join(train_path, f) for f in os.listdir(train_path) if not f.startswith('.')]
test_filenames = [os.path.join(test_path, f) for f in os.listdir(test_path) if not f.startswith('.')]

# aif_batch, lf_batch = input_pipeline(train_filenames, lfsize, patchsize, batchsize, num_crops)
aif_batch, lf_batch = torch_input_pipeline(train_filenames, lfsize, patchsize, batchsize, num_crops)

#forward model
# ray_depths, lf_shear, y = forward_model(aif_batch, lfsize, disp_mult)
ray_depths, lf_shear, y = torch_forward_model(aif_batch, lfsize, disp_mult)

#training losses to minimize
lam_tv = 0.01
lam_dc = 0.005


with tf.name_scope('loss'):
    shear_loss = tf.reduce_mean(tf.abs(lf_shear-lf_batch))
    output_loss = tf.reduce_mean(tf.abs(y-lf_batch))
    tv_loss = lam_tv * tv_loss(ray_depths)
    depth_consistency_loss = lam_dc * depth_consistency_loss(ray_depths, lfsize)
    train_loss = shear_loss + output_loss + tv_loss + depth_consistency_loss


def torch_compute_loss():
    shear_loss = torch.mean(torch.abs(lf_shear-lf_batch))
    output_loss = torch.mean(torch.abs(y-lf_batch))
    tv_loss = lam_tv * torch_tv_loss(ray_depths)
    depth_consistency_loss = lam_dc * torch_depth_consistency_loss(ray_depths, lfsize)
    train_loss = shear_loss + output_loss + tv_loss + depth_consistency_loss
    return train_loss


with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss)

def train():
    # instantiate model

    optimizer = torch.optim.Adam(, lr=learning_rate)
    for i in range(train_iters):
        # call forward
        # call loss
        # call backward
        # call optimizer
    
    
    return

#tensorboard summaries
tf.summary.scalar('shear_loss', shear_loss)
tf.summary.scalar('output_loss', output_loss)
tf.summary.scalar('tv_loss', tv_loss)
tf.summary.scalar('depth_consistency_loss', depth_consistency_loss)
tf.summary.scalar('train_loss', train_loss)

tf.summary.histogram('ray_depths', ray_depths)

tf.summary.image('input_image', aif_batch)
tf.summary.image('lf_shear', tf.reshape(tf.transpose(lf_shear, perm=[0, 3, 1, 4, 2, 5]),
                                        [batchsize, patchsize[0]*lfsize[2], patchsize[1]*lfsize[3], 3]))
tf.summary.image('lf_output', tf.reshape(tf.transpose(y, perm=[0, 3, 1, 4, 2, 5]),
                                        [batchsize, patchsize[0]*lfsize[2], patchsize[1]*lfsize[3], 3]))
tf.summary.image('ray_depths', tf.reshape(tf.transpose(ray_depths, perm=[0, 3, 1, 4, 2]),
                                        [batchsize, patchsize[0]*lfsize[2], patchsize[1]*lfsize[3], 1]))

merged = tf.summary.merge_all()

logdir = 'logs/train/' #path to store logs
checkpointdir = 'checkpoints/' #path to store checkpoints

# with tf.Session() as sess:
#     train_writer = tf.summary.FileWriter(logdir, sess.graph)
#     saver = tf.train.Saver()

#     sess.run(tf.local_variables_initializer())
#     sess.run(tf.global_variables_initializer()) #initialize variables

#     coord = tf.train.Coordinator() #coordinator for input queue threads
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord) #start input queue threads

#     for i in range(train_iters):
#         #training training step
#         _ = sess.run(train_step)
#         #save training summaries
#         if (i+1) % 1 == 0: #can change the frequency of writing summaries if desired
#             print('training step: ', i)
#             trainsummary = sess.run(merged)
#             train_writer.add_summary(trainsummary, i)
#         #save checkpoint
#         if (i+1) % 4000 == 0:
#             saver.save(sess, checkpointdir + 'model.ckpt', global_step=i)

#     #cleanup
#     train_writer.close()

#     coord.request_stop()
#     coord.join(threads)