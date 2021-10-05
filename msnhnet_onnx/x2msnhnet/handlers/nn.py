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
        useBias = len(node.input_tensor_names) == 3
        x = tensor_dict[node.input_tensor_names[0]]
        weight = tensor_dict[node.input_tensor_names[1]]
        msnhnet_weights.extend(weight.flatten().tolist())
        if useBias:
            bias = tensor_dict[node.input_tensor_names[2]]
            msnhnet_weights.extend(bias.flatten().tolist())

        msnhnet_params.extend("conv:\n")
        msnhnet_params.extend(f"  filters: {weight.shape[0]} \n")
        msnhnet_params.extend(f"  kSizeX: {node.attrs['kernel_shape'][0]}\n")
        msnhnet_params.extend(f"  kSizeY: {node.attrs['kernel_shape'][1]}\n")
        msnhnet_params.extend(f"  paddingX: {node.attrs['pads'][0]}\n")
        msnhnet_params.extend(f"  paddingY: {node.attrs['pads'][1]}\n")
        msnhnet_params.extend(f"  strideX: {node.attrs['strides'][0]}\n")
        msnhnet_params.extend(f"  strideY: {node.attrs['strides'][1]}\n")
        msnhnet_params.extend(f"  dilationX: {node.attrs['dilations'][0]}\n")
        msnhnet_params.extend(f"  dilationY: {node.attrs['dilations'][1]}\n")
        msnhnet_params.extend(f"  groups: {node.attrs['group']}\n")
        msnhnet_params.extend(f"  useBias: {useBias}\n")
        return

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)



@onnx_op("Relu")
class Relu(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        msnhnet_params.extend(f"act:\n")
        msnhnet_params.extend(f"  activation: relu\n")
        return

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_14(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("MaxPool")
class MaxPool(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        msnhnet_params.extend(f"maxpool:\n")
        
        kernel_shape = node.attrs["kernel_shape"]
        spatial_size = len(kernel_shape)
        strides = node.attrs.get("strides", [1] * spatial_size)
        dilations = node.attrs.get("dilations", [1] * spatial_size)
        ceil_mode = node.attrs.get("ceil_mode")
        pads = node.attrs.get("auto_pad", "NOTSET")
        if pads == "NOTSET":
            pads = node.attrs.get("pads", [0] * spatial_size * 2)
            pads = np.reshape(pads, [2, spatial_size]).T.tolist()
            pads = [[0, 0], [0, 0]] + pads
        
        msnhnet_params.extend(f"  kSizeX: {kernel_shape[0]}\n")
        msnhnet_params.extend(f"  kSizeY: {kernel_shape[1]}\n")
        msnhnet_params.extend(f"  paddingX: {pads[2][0]}\n")
        msnhnet_params.extend(f"  paddingY: {pads[3][0]}\n")
        msnhnet_params.extend(f"  strideX: {strides[0]}\n")
        msnhnet_params.extend(f"  strideY: {strides[1]}\n")
        msnhnet_params.extend(f"  ceilMode: {ceil_mode}\n")
        return

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_8(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_10(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("AveragePool")
class AveragePool(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        msnhnet_params.extend(f"localavgpool:\n")

        kernel_shape = node.attrs["kernel_shape"]
        spatial_size = len(kernel_shape)
        strides = node.attrs.get("strides", [1] * spatial_size)
        dilations = node.attrs.get("dilations", [1] * spatial_size)
        ceil_mode = node.attrs.get("ceil_mode")
        pads = node.attrs.get("auto_pad", "NOTSET")
        if pads == "NOTSET":
            pads = node.attrs.get("pads", [0] * spatial_size * 2)
            pads = np.reshape(pads, [2, spatial_size]).T.tolist()
            pads = [[0, 0], [0, 0]] + pads
        
        msnhnet_params.extend(f"  kSizeX: {kernel_shape[0]}\n")
        msnhnet_params.extend(f"  kSizeY: {kernel_shape[1]}\n")
        msnhnet_params.extend(f"  paddingX: {pads[2][0]}\n")
        msnhnet_params.extend(f"  paddingY: {pads[3][0]}\n")
        msnhnet_params.extend(f"  strideX: {strides[0]}\n")
        msnhnet_params.extend(f"  strideY: {strides[1]}\n")
        msnhnet_params.extend(f"  ceilMode: {ceil_mode}\n")
        return

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_7(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_10(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Flatten")
class Flatten(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        return

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

@onnx_op("Gemm")
class Gemm(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        B = tensor_dict[node.input_tensor_names[1]]
        useBias = False
        if len(node.input_tensor_names) == 3:
            useBias = True
            C = tensor_dict[node.input_tensor_names[2]]
        alpha = node.attrs.get("alpha", 1.0)
        beta = node.attrs.get("beta", 1.0)
        transA = node.attrs.get("transA", 0)
        transB = node.attrs.get("transB", 0)

        if transB == True:
            msnhnet_weights.extend(B.T.flatten().tolist())
        else:
            msnhnet_weights.extend(B.flatten().tolist())
        
        if useBias:
            msnhnet_weights.extend(C.flatten().tolist())
        
        msnhnet_params.extend(f"connect:\n")
        output = B.shape[0]
        msnhnet_params.extend(f"  output: {output}\n")
        msnhnet_params.extend(f"  useBias: {useBias}\n")
        return

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_9(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

