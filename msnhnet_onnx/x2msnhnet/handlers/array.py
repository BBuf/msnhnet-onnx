import string
import random
import operator
from functools import reduce

import numpy as np

from msnhnet_onnx.x2msnhnet.handler import BackendHandler
from msnhnet_onnx.x2msnhnet.handler import onnx_op
from msnhnet_onnx.x2msnhnet.handler import msnhnet_weights, msnhnet_params

@onnx_op("Add")
class Add(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        
        return

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
