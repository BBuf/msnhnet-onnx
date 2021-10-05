import string
import random
import operator
from functools import reduce

import numpy as np

from msnhnet_onnx.x2msnhnet.handler import BackendHandler
from msnhnet_onnx.x2msnhnet.handler import onnx_op
from msnhnet_onnx.x2msnhnet.handler import msnhnet_weights, msnhnet_params

@onnx_op("Conv")
class Conv(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        print(len(tensor_dict))
        useBias = len(node.input_tensor_names) == 3
        weight = tensor_dict[node.input_tensor_names[1]]
        msnhnet_weights.extend(weight.flatten().tolist())
        if useBias:
            bias = tensor_dict[node.input_tensor_names[2]]
            msnhnet_weights.extend(bias.flatten().tolist())
        
        msnhnet_params.extend("conv:\n")


        return
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)



