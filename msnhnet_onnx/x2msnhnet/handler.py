from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import inspect
import os
import shutil

from onnx import defs

class BackendHandler:
    """
  All operator handler MUST put decorator @onnx_op to register corresponding op.
  """

    ONNX_OP = None

    DOMAIN = defs.ONNX_DOMAIN
    VERSION = 0
    SINCE_VERSION = 0
    PARTIAL_SUPPORT = False
    PS_DESCRIPTION = ""
    MSNHNET_WEIGHTS = []
    MSNHNET_PARAMS = []
    OP_OUTPUS = []

    @classmethod
    def check_cls(cls):
        if not cls.ONNX_OP:
            common.logger.warning(
                "{} doesn't have ONNX_OP. "
                "Please use BackendHandler.onnx_op decorator to register ONNX_OP.".format(
                    cls.__name__
                )
            )

    @classmethod
    def handle(cls, node, tensor_dict, **kwargs):
        """ Main method in handler. It will find corresponding versioned handle method,
        whose name format is `version_%d`. So prefix `version_` is reserved in onnx-msnhnet.
        DON'T use it for other purpose.

        :param node: NodeProto for backend.
        :param kwargs: Other args.
        :return: MsnhNetNode for backend.
        """
        ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
        if ver_handle:
            return ver_handle(node, tensor_dict, **kwargs)
        raise ValueError(
            'node "{}" of version {} is not supported'.format(
                node.op_type, cls.SINCE_VERSION
            )
        )
        return None

    @classmethod
    def get_versions(cls):
        """ Get all support versions.

    :return: Version list.
    """
        versions = []
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("version_"):
                versions.append(int(k.replace("version_", "")))
        return versions

    @staticmethod
    def onnx_op(op):
        return BackendHandler.property_register("ONNX_OP", op)


    @staticmethod
    def domain(d):
        return BackendHandler.property_register("DOMAIN", d)

    @staticmethod
    def partial_support(ps):
        return BackendHandler.property_register("PARTIAL_SUPPORT", ps)

    @staticmethod
    def ps_description(psd):
        return BackendHandler.property_register("PS_DESCRIPTION", psd)

    @staticmethod
    def property_register(name, value):
        def deco(cls):
            setattr(cls, name, value)
            return cls

        return deco

    FLOW_FUNC = None
    WEIGHT_SAVE_DIR = None

    @classmethod
    def copy_variable_file(cls, src_var_name, dst_var_name):
        dst_dir_name = os.path.join(cls.WEIGHT_SAVE_DIR, dst_var_name)
        if not os.path.exists(dst_dir_name):
            os.makedirs(dst_dir_name)
        shutil.copyfile(
            os.path.join(cls.WEIGHT_SAVE_DIR, src_var_name, "out"),
            os.path.join(dst_dir_name, "out"),
        )

    @classmethod
    def get_attrs_processor_param(cls):
        """ Get param for attrs processor.

    :return: Dict.
    """
        return {}

    @classmethod
    def _process_attrs(cls, attrs):
        """ Private method for processing attrs.
        Param for this processor got from `get_attrs_processor_param`.
        Param is dict contains two key: `default` and `raname`.
        First add default value to attrs if key does not exist.
        Second rename key to new key.

        For example:
        attrs = {"keep_dims": True}
        param = {"default": {"axis": 1},
                "rename": {"keep_dims": "keepdims"}}

        processed_attrs = {"axis": "1", "keepdims": True}

        :param attrs: Process target attrs.
        :return: Processed attrs.
        """
        param = {"rename": {}, "default": {}}
        param.update(cls.get_attrs_processor_param())

        for k, v in param["default"].items():
            attrs.setdefault(k, v)

        for k, new_k in param["rename"].items():
            if k in attrs:
                attrs[new_k] = attrs.pop(k)

        return attrs

domain = BackendHandler.domain
onnx_op = BackendHandler.onnx_op
partial_support = BackendHandler.partial_support
ps_description = BackendHandler.ps_description
msnhnet_weights = BackendHandler.MSNHNET_WEIGHTS
msnhnet_params = BackendHandler.MSNHNET_PARAMS

