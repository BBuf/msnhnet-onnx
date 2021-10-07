import string
import random
import operator
from functools import reduce

import numpy as np

from msnhnet_onnx.x2msnhnet.handler import BackendHandler
from msnhnet_onnx.x2msnhnet.handler import onnx_op
from msnhnet_onnx.x2msnhnet.handler import msnhnet_weights, msnhnet_params, msnhnet_layer_ids, msnhnet_input_layer_shape
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

@onnx_op("Clip")
class Clip(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        
        if cls.SINCE_VERSION < 11:
            # min/max were required and passed as attributes
            clip_value_min = node.attrs.get("min", None)
            clip_value_max = node.attrs.get("max", None)
        else:
            # min/max are optional and passed as input_tensor_names
            init_dict = kwargs["init_dict"]
            clip_value_min = (
                init_dict[node.input_tensor_names[1]].item()
                if len(node.input_tensor_names) > 1 and node.input_tensor_names[1] != ""
                else None
            )
            clip_value_max = (
                init_dict[node.input_tensor_names[2]].item()
                if len(node.input_tensor_names) > 2 and node.input_tensor_names[2] != ""
                else None
            )

        if clip_value_min == 0 and clip_value_max == 6.0:
            msnhnet_params.extend("\n\n")
            msnhnet_params.extend(f"act:\n")
            msnhnet_params.extend(f"  activation: relu6\n")

            for i in range(len(node.output_tensor_names)):
                msnhnet_layer_ids[node.output_tensor_names[i]] = gv.msnhnet_layer_cnt
                gv.msnhnet_layer_cnt += 1
            return
        else:
            raise NotImplementedError(f'MsnhNet not support clip op now!')    
            return

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_6(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_12(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

def buildView(dim0, dim1, dim2):
    msnhnet_params.extend("\n\n")
    msnhnet_params.extend(f"view:\n")
    msnhnet_params.extend(f"  dim0: {dim0}\n")
    msnhnet_params.extend(f"  dim1: {dim0}\n")
    msnhnet_params.extend(f"  dim2: {dim0}\n")

@onnx_op("Reshape")
class Reshape(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        init_dict = kwargs["init_dict"]
        x = msnhnet_input_layer_shape[node.input_tensor_names[0]]
        if cls.SINCE_VERSION == 1:
            shape = node.attrs["shape"]
        else:  # since_version >= 5
            shape = init_dict[node.input_tensor_names[1]]
            node.attrs["shape"] = shape.tolist()
            del node.input_tensor_names[1]

        dataSize = x[1] * x[2] * x[3]
        shape = node.attrs["shape"]
        if len(shape) == 2:
            if shape[0] == -1 and shape[1] != -1:
                if dataSize % shape[1] != 0:
                    raise NotImplementedError("params error")
                dim1 = dataSize/shape[1]
                dim2 = shape[1]
                buildView(1,dim1,dim2)
            elif shape[0] != -1 and shape[1] == -1:
                if dataSize % shape[1] != 0:
                    raise NotImplementedError("params error")
                dim1 = shape[0]
                dim2 = dataSize/shape[0]
                buildView(1,dim1,dim2)
            elif shape[0] != -1 and shape[1] != -1:
                if dataSize % (shape[1]*shape[0]) != 0:
                    raise NotImplementedError("params error")
                dim1 = shape[0]
                dim2 = shape[1]
                buildView(1,dim1,dim2)
            else:
                raise NotImplementedError("params error")
        if len(shape) == 3:
            if shape[0] == -1 and shape[1] != -1 and shape[2] != -1:
                if dataSize % (shape[1]*shape[2]) != 0:
                    raise NotImplementedError("params error")
                dim0 = dataSize /(shape[1]*shape[2])
                dim1 = shape[1]
                dim2 = shape[2]
                buildView(dim0,dim1,dim2)
            elif shape[0] != -1 and shape[1] == -1 and shape[2] != -1:
                if dataSize % (shape[0]*shape[2]) != 0:
                    raise NotImplementedError("params error")
                dim0 = shape[0]
                dim1 = dataSize/(shape[0]*shape[2])
                dim2 = shape[2]
                buildView(dim0,dim1,dim2)
            elif shape[0] != -1 and shape[1] != -1 and shape[2] == -1:
                if dataSize % (shape[0]*shape[1]) != 0:
                    raise NotImplementedError("params error")
                dim0 = shape[0]
                dim1 = shape[1]
                dim2 = dataSize/(shape[0]*shape[1])
                buildView(dim0,dim1,dim2)
            elif shape[0] != -1 and shape[1] != -1 and shape[2] != -1:
                if dataSize / (shape[0]*shape[1]*shape[2]) != 1:
                    raise NotImplementedError("params error")
                dim0 = shape[0]
                dim1 = shape[1]
                dim2 = shape[2]
                buildView(dim0,dim1,dim2)
        if len(shape) == 4:
            if shape[0] == -1:
                if dataSize/(shape[1]*shape[2]*shape[3])==1 :
                    dim0 = shape[1]
                    dim1 = shape[2]
                    dim2 = shape[3]
                    buildView(dim0,dim1,dim2)
                else:
                    raise NotImplementedError("params error")
            elif shape[0] == 1:
                if shape[1] == -1 and shape[2] != -1 and shape[3] != -1:
                    if dataSize % (shape[1]*shape[2]) != 0:
                        raise NotImplementedError("params error")
                    dim0 = dataSize /(shape[2]*shape[3])
                    dim1 = shape[2]
                    dim2 = shape[3]
                    buildView(dim0,dim1,dim2)
                elif shape[1] != -1 and shape[2] == -1 and shape[3] != -1:
                    if dataSize % (shape[1]*shape[3]) != 0:
                        raise NotImplementedError("params error")
                    dim0 = shape[1]
                    dim1 = dataSize/(shape[1]*shape[3])
                    dim2 = shape[3]
                    buildView(dim0,dim1,dim2)
                elif shape[1] != -1 and shape[2] != -1 and shape[3] == -1:
                    if dataSize % (shape[1]*shape[2]) != 0:
                        raise NotImplementedError("params error")
                    dim0 = shape[1]
                    dim1 = shape[2]
                    dim2 = dataSize/(shape[1]*shape[2])
                    buildView(dim0,dim1,dim2)
                elif shape[1] != -1 and shape[2] != -1 and shape[3] != -1:
                    if dataSize / (shape[1]*shape[2]*shape[3]) != 1:
                        raise NotImplementedError("params error")
                    dim0 = shape[1]
                    dim1 = shape[2]
                    dim2 = shape[3]
                    buildView(dim0,dim1,dim2)
        
        for i in range(len(node.output_tensor_names)):
            msnhnet_layer_ids[node.output_tensor_names[i]] = gv.msnhnet_layer_cnt
            gv.msnhnet_layer_cnt += 1

        return

    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_5(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_14(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
