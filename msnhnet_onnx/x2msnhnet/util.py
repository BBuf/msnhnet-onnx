
from collections import OrderedDict
import tempfile
import os
import shutil

import numpy as np
import onnxruntime as ort
import onnx
import torch
import paddle
import tensorflow as tf

from msnhnet_onnx.x2msnhnet.handler import msnhnet_params, msnhnet_weights
from msnhnet_onnx.x2msnhnet.onnx2msnhnet import from_onnx, from_pytorch, from_paddle, from_tensorflow2

def load_pytorch_module_and_check(
    pt_module_class,
    input_size=None,
    input_min_val=0.0,
    input_max_val=1.0,
    train_flag=False,
    msnhnet_weight_dir="/tmp/msnhnet",
    msnhnet_params_flag=False,
):
    if input_size is None:
        input_size = (2, 4, 3, 5)
    pt_module = pt_module_class()

    model_weight_save_dir = msnhnet_weight_dir
    x = np.ones(input_size)

    y = from_pytorch(
                pt_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )

def load_paddle_module_and_check(
    pd_module_class,
    input_size=None,
    input_min_val=0.0,
    input_max_val=1.0,
    train_flag=False,
    msnhnet_weight_dir="/tmp/msnhnet",
    msnhnet_params_flag = False, 
):
    if input_size is None:
        input_size = (2, 4, 3, 5)
    pd_module = pd_module_class()

    model_weight_save_dir = msnhnet_weight_dir
    x = np.ones(input_size)

    y = from_paddle(
                pd_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )


def load_tensorflow2_module_and_check(
    tf_module_class,
    input_size=None,
    input_min_val=0.0,
    input_max_val=1.0,
    train_flag=False,
    msnhnet_weight_dir="/tmp/msnhnet",
    msnhnet_params_flag = False, 
):
    if input_size is None:
        input_size = (2, 4, 3, 5)
    tf_module = tf_module_class()

    model_weight_save_dir = msnhnet_weight_dir
    x = np.ones(input_size)

    y = from_tensorflow2(
                tf_module,
                x,
                model_weight_dir=model_weight_save_dir,
                do_onnxsim=True,
                train_flag=train_flag,
            )

