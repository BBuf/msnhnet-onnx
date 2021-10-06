from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass

from struct import pack
import copy
from onnx import defs
from onnx import numpy_helper
from onnx.backend.base import Backend
from onnx.backend.base import Device
from onnx.backend.base import namedtupledict
from onnx.helper import make_opsetid
from onnx import numpy_helper

from msnhnet_onnx import util
from msnhnet_onnx.x2msnhnet.handler import BackendHandler

from msnhnet_onnx.x2msnhnet.handlers import *
from msnhnet_onnx.onnx_wrapper import Node as OnnxNode
from msnhnet_onnx.x2msnhnet.handler import msnhnet_params, msnhnet_weights
import io
import tempfile
import os
import shutil
import numpy as np
import onnx
import torch
import paddle
import tensorflow as tf
import tf2onnx
import logging
import onnxoptimizer

try:
    import onnxsim

    has_onnxsim = True
except ImportError:
    has_onnxsim = False

logger = logging.getLogger(__name__)

init_weight_dict = {}

def from_onnx(
    onnx_model: onnx.ModelProto, inputs, model_weight_dir="/tmp/tmp", do_onnxsim=True, from_tf2=False, from_paddle=False, from_pytorch=False, 
):
    # msnhnet_params = []
    # msnhnet_weights = []
    input_names = [x.name for x in onnx_model.graph.input]
    if type(inputs) is not dict:
        assert (
            len(input_names) == 1
        ), "Please use input dict if the model has multiple inputs"
        inputs = {input_names[0]: inputs}
    if do_onnxsim and has_onnxsim:
        dict(zip(input_names, [x.shape for x in inputs.values()]))
        onnx_model, _ = onnxsim.simplify(
            onnx_model,
            skip_fuse_bn=False,
            skip_shape_inference=False,
            input_shapes=dict(zip(input_names, [x.shape for x in inputs.values()])),
        )
    elif do_onnxsim:
        logger.info(
            "We recommend installing onnx-simplifier so that MsnhNet can remove the redundant ONNX nodes"
        )
    
    initializer_name = []
    if from_tf2:
        for x in onnx_model.graph.input:
            x.name = x.name.replace('/', '_')
            x.name = x.name.replace(':', '_')
        for i, node in enumerate(onnx_model.graph.node):
            node.name = node.name.replace('/', '_')
            node.name = node.name.replace(':', '_')
            for j in range(len(node.input)):
                node.input[j] = node.input[j].replace('/', '_')
                node.input[j] = node.input[j].replace(':', '_')
            for j in range(len(node.output)):
                node.output[j] = node.output[j].replace('/', '_')
                node.output[j] = node.output[j].replace(':', '_')
        for x in onnx_model.graph.initializer:
            x.name = x.name.replace('/', '_')
            x.name = x.name.replace(':', '_')
            initializer_name.append(x.name)
        # to solve tf batchnorm without scale params
        delete_node_name = []
        for i, node in enumerate(onnx_model.graph.node):
            if node.op_type == "BatchNormalization":
                if node.input[1] in initializer_name:
                    pass
                else:
                    delete_node_name.append(node.input[1])
        
        for i, x in enumerate(onnx_model.graph.input):
            if x.name in delete_node_name:
                tensor_dim = onnx_model.graph.input[i].type.tensor_type.shape.dim
                new_bn_value = []
                for j in range(int(tensor_dim[0].dim_value)):
                    new_bn_value.append(1.0)
                new_bn_scale_node = onnx.helper.make_tensor(name=x.name, data_type=onnx.TensorProto.FLOAT, dims=(int(tensor_dim[0].dim_value),), vals=new_bn_value)
                onnx_model.graph.initializer.extend([new_bn_scale_node])
        
        for x in onnx_model.graph.input:
            if x.name in delete_node_name:
                onnx_model.graph.input.remove(x)

    # to solve paddlepaddle2msnhnet initializer rename bug
    if from_paddle == True:
        
        graph_input_name = {}
        graph_initializer_name = []
        for x in onnx_model.graph.initializer:
            graph_initializer_name.append(x.name)

        for i, node in enumerate(onnx_model.graph.node):
            # node_cp = node
            node_cp = copy.deepcopy(node)
            for j in range(len(node.input)):
                if node.input[j] in graph_initializer_name:
                    node_cp.input[j] = node.name + "_" + node.input[j]
                    graph_input_name[node_cp.input[j]] = node.input[j]
            onnx_model.graph.node.remove(node)
            onnx_model.graph.node.insert(i, node_cp)
        
        extend_op = []
        for k, v in graph_input_name.items():
            for x in onnx_model.graph.initializer:
                base_name = x.name
                if x.name == v:
                    x.name = k
                    for k2, v2 in graph_input_name.items():
                        if v2 == base_name and k2 != k:
                            x_cp = copy.deepcopy(x)
                            x_cp.name = k2
                            extend_op.append(x_cp)
            for x in onnx_model.graph.input:
                if x.name == v:
                    onnx_model.graph.input.remove(x)
        for x in extend_op:
            onnx_model.graph.initializer.extend([x])
    
    # for code gen
    for x in onnx_model.graph.input:
            x.name = x.name.replace('.', '_')
            x.name = x.name.replace('/', '_')
            x.name = x.name.replace(':', '_')
    for i, node in enumerate(onnx_model.graph.node):
        node.name = node.name.replace('.', '_')
        node.name = node.name.replace('/', '_')
        node.name = node.name.replace(':', '_')
        for j in range(len(node.input)):
            node.input[j] = node.input[j].replace('.', '_')
            node.input[j] = node.input[j].replace('/', '_')
            node.input[j] = node.input[j].replace(':', '_')
        for j in range(len(node.output)):
            node.output[j] = node.output[j].replace('.', '_')
            node.output[j] = node.output[j].replace('/', '_')
            node.output[j] = node.output[j].replace(':', '_')
    for x in onnx_model.graph.initializer:
        x.name = x.name.replace('.', '_')
        x.name = x.name.replace('/', '_')
        x.name = x.name.replace(':', '_')
    for x in onnx_model.graph.output:
        x.name = x.name.replace('.', '_')
        x.name = x.name.replace('/', '_')
        x.name = x.name.replace(':', '_')
    
    graph_initializer_name = []
    for x in onnx_model.graph.initializer:
        graph_initializer_name.append(x.name)
    graph_name_dict = {}
    rename_set = []
    for i, node in enumerate(onnx_model.graph.node):
        # node_cp = node
        node_cp = copy.deepcopy(node)
        if node.name == '':
            cnt = 0
            while True:
                node.name = node.op_type + '_{}'.format(cnt)
                if node.name in rename_set:
                    pass
                else:
                    rename_set.append(node.name)
                    break
                cnt = cnt + 1
        for j in range(len(node.input)):
            if node.input[j] == 'x_0':
                node_cp.input[j] = node.input[j]
            elif node.input[j] in graph_name_dict:
                node_cp.input[j] = graph_name_dict[node.input[j]]
            else:
                if node.op_type == "Clip" and (node.input[j] not in graph_initializer_name):
                    pass
                else:
                    node_cp.input[j] = node.name.lower() + '_input_{}'.format(j)
                    graph_name_dict[node.input[j]] = node_cp.input[j]
        for j in range(len(node.output)):
            if node.output[j] in graph_name_dict:
                node_cp.output[j] = graph_name_dict[node.output[j]]
            else:
                node_cp.output[j] = node.name.lower() + '_output_{}'.format(j)
                graph_name_dict[node.output[j]] = node_cp.output[j]
        
        onnx_model.graph.node.remove(node)
        onnx_model.graph.node.insert(i, node_cp)

    for x in onnx_model.graph.input:
        if x.name in graph_name_dict:
            x.name = graph_name_dict[x.name]
    for x in onnx_model.graph.output:
        if x.name in graph_name_dict:
            x.name = graph_name_dict[x.name]
    for x in onnx_model.graph.initializer:
        if x.name in graph_name_dict:
            x.name = graph_name_dict[x.name]
    
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    
    # to save onnx model after onnx_simplifier
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")
    onnx.save(onnx_model, "/tmp/simp.onnx")

    for x in onnx_model.graph.initializer:
        init_weight_dict[x.name] = numpy_helper.to_array(x)

    d = prepare(onnx_model, blob_dict=inputs)

    if not os.path.exists(model_weight_dir):
        os.makedirs(model_weight_dir)
    
    with open(os.path.join(model_weight_dir, "model.msnhnet"), "w") as temp_file:
        for x in msnhnet_params:
            temp_file.write("%s" % x)
    
    with open(os.path.join(model_weight_dir, "model.msnhbin"), "wb") as temp_file:
        for x in msnhnet_weights:
            temp_file.write(pack('f', x))

    output_names = [x.name for x in onnx_model.graph.output]
    if len(output_names) == 1:
        return d[output_names[0]]

    return {output_name: d[output_name] for output_name in output_names}
    


def from_pytorch(
    torch_model, inputs, model_weight_dir="/tmp", do_onnxsim=True, train_flag=True
):
    if type(inputs) is not list:
        inputs = [inputs]
    input_names = ["x_{}".format(i) for i in range(len(inputs))]

    assert len(inputs[0].shape) == 4

    msnhnet_params.extend(f"config:\n")
    msnhnet_params.extend(f"   batch: {inputs[0].shape[0]}\n")
    msnhnet_params.extend(f"   height: {inputs[0].shape[2]}\n")
    msnhnet_params.extend(f"   width: {inputs[0].shape[3]}\n")
    msnhnet_params.extend(f"   channels: {inputs[0].shape[1]}\n")

    torch_model = torch_model.to("cpu")

    f = io.BytesIO()
    torch.onnx.export(
        torch_model,
        tuple([torch.zeros(ipt.shape) for ipt in inputs]),
        f,
        input_names=input_names,
        opset_version=12,
        training=train_flag,
    )
    model_str = f.getvalue()
    onnx_model = onnx.load_model_from_string(model_str)
    return from_onnx(
        onnx_model,
        dict(zip(input_names, inputs)),
        model_weight_dir=model_weight_dir,
        do_onnxsim=do_onnxsim,
        from_pytorch=True,
    )


def from_paddle(
    paddle_model, inputs, model_weight_dir="/tmp", do_onnxsim=True, train_flag=True
):
    input_names = "x_0"
    paddle_model.eval()
    input_spec = paddle.static.InputSpec(
        shape=inputs.shape, dtype="float32", name=input_names
    )

    assert len(inputs.shape) == 4

    msnhnet_params.extend(f"config:\n")
    msnhnet_params.extend(f"   batch: {inputs.shape[0]}\n")
    msnhnet_params.extend(f"   height: {inputs.shape[2]}\n")
    msnhnet_params.extend(f"   width: {inputs.shape[3]}\n")
    msnhnet_params.extend(f"   channels: {inputs.shape[1]}\n")

    mode_str = "/tmp/tmp"

    paddle.onnx.export(
        paddle_model,
        mode_str,
        input_spec=[input_spec],
        opset_version=12,
        enable_onnx_checker=True,
    )

    onnx_model = onnx.load(str(mode_str + ".onnx"))

    return from_onnx(
        onnx_model,
        dict(zip([input_names], [inputs])),
        model_weight_dir=model_weight_dir,
        do_onnxsim=do_onnxsim,
        from_paddle=True,
    )


def from_tensorflow2(
    tf_model, inputs, model_weight_dir="/tmp", do_onnxsim=True, train_flag=True
):
    input_names = "x_0"

    assert len(inputs.shape) == 4

    msnhnet_params.extend(f"config:\n")
    msnhnet_params.extend(f"   batch: {inputs.shape[0]}\n")
    msnhnet_params.extend(f"   height: {inputs.shape[2]}\n")
    msnhnet_params.extend(f"   width: {inputs.shape[3]}\n")
    msnhnet_params.extend(f"   channels: {inputs.shape[1]}\n")
    
    # input_spec = paddle.static.InputSpec(
    #     shape=inputs.shape, dtype="float32", name=input_names
    # )
    spec = (tf.TensorSpec(inputs.shape, tf.float32, name=input_names),)

    mode_str = "/tmp/tmp.onnx"

    model_proto, _ = tf2onnx.convert.from_keras(
        tf_model, input_signature=spec, opset=11, output_path=mode_str
    )

    return from_onnx(
        model_proto,
        dict(zip([input_names], [inputs])),
        model_weight_dir=model_weight_dir,
        do_onnxsim=do_onnxsim,
        from_tf2=True,
    )


def get_all_backend_handlers(opset_dict):
    """ Get a dict of all backend handler classes.
  e.g. {'domain': {'Abs': Abs handler class}, ...}, }.
  :param opset_dict: A dict of opset. e.g. {'domain': version, ...}
  :return: Dict.
  """
    handlers = {}
    for handler in BackendHandler.__subclasses__():
        handler.check_cls()

        domain = handler.DOMAIN
        version = opset_dict[domain]
        handler.VERSION = version

        since_version = 1
        if defs.has(handler.ONNX_OP, domain=handler.DOMAIN):
            try:
                since_version = defs.get_schema(
                    handler.ONNX_OP,
                    domain=handler.DOMAIN,
                    max_inclusive_version=version,
                ).since_version
            except RuntimeError:
                logger.info(
                    "Fail to get since_version of {} in domain `{}` "
                    "with max_inclusive_version={}. Set to 1.".format(
                        handler.ONNX_OP, handler.DOMAIN, version
                    )
                )
        else:
            logger.info(
                "Unknown op {} in domain `{}`.".format(
                    handler.ONNX_OP, handler.DOMAIN or "ai.onnx"
                )
            )
        handler.SINCE_VERSION = since_version
        handlers.setdefault(domain, {})[handler.ONNX_OP] = handler
    return handlers


class MsnhnetBackend(Backend):
    """ Msnhnet Backend for ONNX
    """

    @classmethod
    def prepare(
        cls,
        model,
        device="CPU",
        strict=True,
        logging_level="INFO",
        blob_dict=None,
        **kwargs
    ):
        """Prepare an ONNX model for MsnhNet Backend.
    :param model: The ONNX model to be converted.
    :param device: The device to execute this model on.
    :param strict: Whether to enforce semantic equivalence between the original model
      and the converted msnhnet model, defaults to True (yes, enforce semantic equivalence).
      Changing to False is strongly discouraged.
      Currently, the strict flag only affects the behavior of MaxPool and AveragePool ops.
    :param logging_level: The logging level, default is INFO. Change it to DEBUG
      to see more conversion details or to WARNING to see less
    :returns: The variable dict of the converted msnhnet model
    """
        super(MsnhnetBackend, cls).prepare(model, device, **kwargs)
        logger.setLevel(logging_level)

        return cls.onnx_model_to_msnhnet(model, strict, blob_dict=blob_dict)

    @classmethod
    def onnx_model_to_msnhnet(cls, model, strict, blob_dict=None):
        """ Convert ONNX model to MsnhNet.
    :param model: ONNX ModelProto object.
    :param strict: whether to enforce semantic equivalence between the original model
      and the converted msnhnet model.
    :return: The variable dict of the converted msnhnet model
    """

        # Models with IR_VERSION less than 3 does not have opset_import set.
        # We default to minimum opset, this behavior is consistent with
        # onnx checker.
        # c.f. https://github.com/onnx/onnx/blob/427ac0c1b792363d373e3d7e4eef97fa46458420/onnx/checker.cc#L478
        if model.ir_version < 3:
            opset_import = [make_opsetid(defs.ONNX_DOMAIN, 1)]
        else:
            opset_import = model.opset_import
        return cls._onnx_graph_to_msnhnet(
            model.graph, opset_import, strict, blob_dict=blob_dict
        )

    @classmethod
    def _onnx_graph_to_msnhnet(cls, graph_def, opset, strict, blob_dict=None):
        """ Convert ONNX graph to msnhnet.
        :param graph_def: ONNX GraphProto object.
        :param opset: ONNX OperatorSetIdProto list.
        :param strict: whether to enforce semantic equivalence between the original model
          and the converted msnhnet.
        :param blob_dict: {name: msnhnet_blob}, the inputs of onnx graph will be populated with msnhnet_blob with the same name
        :return: The variable dict of the converted msnhnet model
        """
        if blob_dict is None:
            blob_dict = {}
        handlers = cls._get_handlers(opset)

        # initializer: TensorProtos representing the values to initialize
        # a given tensor.
        # initialized: A list of names of the initialized tensors.
        
        if graph_def.initializer:
            input_dict_items = cls._onnx_initializer_to_input_dict_items(
                graph_def.initializer
            )
            initialized = {
                init.name: onnx.numpy_helper.to_array(init)
                for init in graph_def.initializer
            }
        else:
            input_dict_items = []
            initialized = {}

        for node in graph_def.node:
            node = OnnxNode(node)
            if node.op_type == "Constant":
                initialized[node.output_tensor_names[0]] = numpy_helper.to_array(
                    node.attrs["value"]
                )

        # creating placeholders for currently unknown inputs
        for value_info in graph_def.input:
            if value_info.name in initialized:
                continue
            shape = list(
                d.dim_value if (d.dim_value > 0 and d.dim_param == "") else None
                for d in value_info.type.tensor_type.shape.dim
            )
            if value_info.name not in blob_dict:
                raise NotImplementedError("no blob named {}".format(value_info.name))
            input_dict_items.append((value_info.name, blob_dict[value_info.name]))

        # tensor dict: this dictionary is a map from variable names
        # to the latest produced msnhnet variables of the given name.
        # This dictionary will get updated as we build the graph to
        # record the names of newly produced tensors.
        tensor_dict = dict(input_dict_items)
        # Since tensor dict may be updated, we need to keep a copy
        # of the original input dict where we track the earliest
        # defined tensors so we can have access to the placeholders
        # to feed in input tensors when we run the graph.
        input_dict = dict(input_dict_items)

        for node in graph_def.node:
            onnx_node = OnnxNode(node)
            output_ops = cls._onnx_node_to_msnhnet_op(
                onnx_node,
                tensor_dict,
                initialized,
                handlers,
                opset=opset,
                strict=strict,
            )
            curr_node_output_map = dict(zip(onnx_node.output_tensor_names, output_ops))
            tensor_dict.update(curr_node_output_map)
        return tensor_dict

    @classmethod
    def _onnx_initializer_to_input_dict_items(cls, initializer):
        """ Convert ONNX graph initializer to input dict items.
    :param initializer: ONNX graph initializer, list of TensorProto.
    :return: List of input dict items.
    """

        def get_msnhnet_shape(shape):
            if len(shape) == 0:
                return (1,)
            return shape

        return [
            (
                init.name,
                # flow.get_variable(
                #     name=init.name,
                #     shape=get_flow_shape(list(init.dims)),
                #     initializer=flow.zeros_initializer(),
                #     trainable=True,
                #     dtype=util.Onnx2FlowDtype(init.data_type),
                # ),
                init_weight_dict[init.name],
            )
            for init in initializer
        ]

    @classmethod
    def _onnx_node_to_msnhnet_op(
        cls, node, tensor_dict, init_dict, handlers=None, opset=None, strict=True
    ): 
        """
    Convert onnx node to msnhnet op.
    Args:
      node: Onnx node object.
      tensor_dict: Tensor dict of graph.
      opset: Opset version of the operator set. Default 0 means using latest version.
      strict: whether to enforce semantic equivalence between the original model
        and the converted msnhnet model, defaults to True (yes, enforce semantic equivalence).
        Changing to False is strongly discouraged.
    Returns:
      msnhnet op
    """
        handlers = handlers or cls._get_handlers(opset)
        handler = handlers[node.domain].get(node.op_type, None)
        if handler:
            output = handler.handle(
                node, tensor_dict, init_dict=init_dict, strict=strict
            )
            if not isinstance(output, (list, tuple)):
                output = [output]
            return output
        else:
            raise ValueError("{} is not supported".format(node.op_type))

    @classmethod
    def _get_handlers(cls, opset):
        """ Get all backend handlers with opset.
    :param opset: ONNX OperatorSetIdProto list.
    :return: All backend handlers.
    """
        opset = opset or [make_opsetid(defs.ONNX_DOMAIN, defs.onnx_opset_version())]
        opset_dict = dict([(o.domain, o.version) for o in opset])
        return get_all_backend_handlers(opset_dict)


prepare = MsnhnetBackend.prepare
