"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import operator
from functools import reduce

import numpy as np

from msnhnet_onnx.x2oneflow.handler import BackendHandler
from msnhnet_onnx.x2oneflow.handler import onnx_op
from msnhnet_onnx.x2oneflow.handler import flow_func
from msnhnet_onnx.x2oneflow.handlers.common import ReductionMixin
import oneflow as flow


@onnx_op("ReduceMean")
@flow_func(flow.math.reduce_mean)
class ReduceMean(ReductionMixin, BackendHandler):
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)


@onnx_op("ReduceMax")
class ReduceMax(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        axes = node.attrs.get("axes")
        keepdims = node.attrs.get("keepdims")
        return flow.math.reduce_max(x, axis=axes, keepdims=bool(keepdims))
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
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

@onnx_op("ReduceMin")
class ReduceMin(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        axes = node.attrs.get("axes")
        keepdims = node.attrs.get("keepdims")
        return flow.math.reduce_min(x, axis=axes, keepdims=bool(keepdims))
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
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



@onnx_op("ReduceProd")
@flow_func(flow.math.reduce_prod)
class ReduceProd(BackendHandler):
    @classmethod
    def _common(cls, node, tensor_dict, **kwargs):
        x = tensor_dict[node.input_tensor_names[0]]
        axes = node.attrs.get("axes")
        keepdims = node.attrs.get("keepdims")
        return flow.math.reduce_prod(x, axis=axes, keepdims=bool(keepdims))
    
    @classmethod
    def version_1(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)

    @classmethod
    def version_11(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
    
    @classmethod
    def version_13(cls, node, tensor_dict, **kwargs):
        return cls._common(node, tensor_dict, **kwargs)
