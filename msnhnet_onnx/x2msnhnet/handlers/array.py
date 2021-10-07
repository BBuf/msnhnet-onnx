import string
import random
import operator
from functools import reduce

import numpy as np

from msnhnet_onnx.x2msnhnet.handler import BackendHandler
from msnhnet_onnx.x2msnhnet.handler import onnx_op
from msnhnet_onnx.x2msnhnet.handler import msnhnet_weights, msnhnet_params, msnhnet_layer_ids
import msnhnet_onnx.x2msnhnet.handlers.global_var as gv
@onnx_op("Add")
class Add(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        layers = ""
        for i in range(len(node.input_tensor_names)):
            layers += str(msnhnet_layer_ids[node.input_tensor_names[i]])
            layers += ", "
        
        msnhnet_params.extend(f"\n\n")
        msnhnet_params.extend(f"route:\n")
        msnhnet_params.extend(f"  layers: {layers}\n")
        msnhnet_params.extend(f"  addModel: 1\n")

        for i in range(len(node.output_tensor_names)):
            msnhnet_layer_ids[node.output_tensor_names[i]] = gv.msnhnet_layer_cnt
            gv.msnhnet_layer_cnt += 1

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

