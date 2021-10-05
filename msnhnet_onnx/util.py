# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# msnhnet_onnx.util - misc utilities for msnhnet_onnx

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import re
import shutil
import tempfile

from google.protobuf import text_format
import numpy as np
import onnx
from onnx import helper, onnx_pb, defs, numpy_helper
import six

from msnhnet_onnx import constants


#
# mapping dtypes from onnx to numpy
#
ONNX_2_NUMPY_DTYPE = {
    onnx_pb.TensorProto.FLOAT: np.float32,
    onnx_pb.TensorProto.FLOAT16: np.float16,
    onnx_pb.TensorProto.DOUBLE: np.float64,
    onnx_pb.TensorProto.INT32: np.int32,
    onnx_pb.TensorProto.INT16: np.int16,
    onnx_pb.TensorProto.INT8: np.int8,
    onnx_pb.TensorProto.UINT8: np.uint8,
    onnx_pb.TensorProto.UINT16: np.uint16,
    onnx_pb.TensorProto.INT64: np.int64,
    onnx_pb.TensorProto.UINT64: np.uint64,
    onnx_pb.TensorProto.BOOL: np.bool,
}

#
#  onnx dtype names
#
ONNX_DTYPE_NAMES = {
    onnx_pb.TensorProto.FLOAT: "float",
    onnx_pb.TensorProto.FLOAT16: "float16",
    onnx_pb.TensorProto.DOUBLE: "double",
    onnx_pb.TensorProto.INT32: "int32",
    onnx_pb.TensorProto.INT16: "int16",
    onnx_pb.TensorProto.INT8: "int8",
    onnx_pb.TensorProto.UINT8: "uint8",
    onnx_pb.TensorProto.UINT16: "uint16",
    onnx_pb.TensorProto.INT64: "int64",
    onnx_pb.TensorProto.STRING: "string",
    onnx_pb.TensorProto.BOOL: "bool",
}


def is_integral_onnx_dtype(dtype):
    return dtype in [
        onnx_pb.TensorProto.INT32,
        onnx_pb.TensorProto.INT16,
        onnx_pb.TensorProto.INT8,
        onnx_pb.TensorProto.UINT8,
        onnx_pb.TensorProto.UINT16,
        onnx_pb.TensorProto.INT64,
    ]


ONNX_UNKNOWN_DIMENSION = -1
ONNX_EMPTY_INPUT = ""



def Numpy2OnnxDtype(np_dtype):
    for onnx_dtype, numpy_dtype in ONNX_2_NUMPY_DTYPE.items():
        if numpy_dtype == np_dtype:
            return onnx_dtype
    raise ValueError("unsupported dtype " + np_dtype + " for mapping")


def Onnx2NumpyDtype(onnx_type):
    return ONNX_2_NUMPY_DTYPE[onnx_type]


def FindOpset(opset):
    """Find opset."""
    if opset is None or opset == 0:
        opset = defs.onnx_opset_version()
        if opset > constants.PREFERRED_OPSET:
            # if we use a newer onnx opset than most runtimes support, default to the one most supported
            opset = constants.PREFERRED_OPSET
    return opset


def MakeSure(bool_val, error_msg, *args):
    if not bool_val:
        raise ValueError("MakeSure failure: " + error_msg % args)


def AreShapesEqual(src, dest):
    """ Check whether 2 shapes are equal. """
    if src is None:
        return dest is None
    if dest is None:
        return src is None

    def is_list_or_tuple(obj):
        return isinstance(obj, (list, tuple))

    MakeSure(is_list_or_tuple(src), "invalid type for src")
    MakeSure(is_list_or_tuple(dest), "invalid type for dest")

    if len(src) != len(dest):
        return False
    return all(i == j for i, j in zip(src, dest))


def get_onnx_version():
    return onnx.__version__


def is_onnx_domain(domain):
    if domain is None or domain == "":
        return True
    return False


def GenerateValidFilename(s):
    return "".join([c if c.isalpha() or c.isdigit() else "_" for c in s])
