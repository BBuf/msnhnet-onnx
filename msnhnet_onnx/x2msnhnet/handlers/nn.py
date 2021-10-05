import string
import random
import operator
from functools import reduce

import numpy as np

from msnhnet_onnx.x2msnhnet.handler import BackendHandler
from msnhnet_onnx.x2msnhnet.handler import onnx_op
from msnhnet_onnx.x2msnhnet.handler import msnhnet_params, msnhnet_weights

@onnx_op("Conv")
class Conv(BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        print('here')
        pass



