# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# msnhnet_onnx.onnx_wrapper - class to manage graph manipulation on top of onnx

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import collections
import copy
import logging
import six
import numpy as np
from os.path import join as pathjoin

from onnx import (
    helper,
    numpy_helper,
    shape_inference,
    OperatorSetIdProto,
    AttributeProto,
    TensorProto,
    onnx_pb,
)

from msnhnet_onnx import util
from msnhnet_onnx.schemas import get_schema

logger = logging.getLogger(__name__)


# pylint: disable=broad-except,protected-access


class Node(object):
    """A Node - wrapper around onnx nodes that we use for graph manipulations."""

    def __init__(self, node, graph=None, skip_conversion=False):
        """Create Node.
        Args:
            node: Onnx node in NodeProto
            graph: Graph() we are part of
        """
        self._op = node
        self.graph = graph
        self._input = list(node.input)
        self._output = list(node.output)
        self.attrs = {}

        if graph is not None:
            graph.set_node_by_name(self)
        # dict to original attributes
        for a in node.attribute:
            attr_val = helper.get_attribute_value(a)
            if isinstance(attr_val, bytes):
                attr_val = attr_val.decode("utf-8")
            self.attrs[a.name] = attr_val
        self._skip_conversion = skip_conversion

    @property
    def input_tensor_names(self):
        return self._input

    @input_tensor_names.setter
    def input_tensor_names(self, val):
        self._input = copy.deepcopy(val)

    @property
    def output_tensor_names(self):
        return copy.deepcopy(self._output)

    @output_tensor_names.setter
    def output_tensor_names(self, val):
        """Set op output. Output should be updated explicitly,
        changing it would require output mapping changed.
        """
        self._GraphCheck()
        for o in self._output:
            del self.graph._output_to_node_name[o]

        self._output = val
        for o in self._output:
            util.MakeSure(
                o not in self.graph._output_to_node_name,
                "output %s already in output mapping",
                o,
            )
            self.graph._output_to_node_name[o] = self.name

    @property
    def input_nodes(self):
        """Input node objects."""
        self._GraphCheck()
        val = [self.graph.get_node_by_output(n) for n in self._input]
        return val

    @property
    def attrs_onnx(self):
        """Return onnx valid attributes"""
        schema = get_schema(self.op_type, self.graph.opset, self.domain)
        if schema is None and not (self.is_const() or self.is_graph_input()):
            logger.debug(
                "Node %s uses non-stardard onnx op <%s, %s>, skip attribute check",
                self.name,
                self.domain,
                self.op_type,
            )
        onnx_attrs = {}
        for name, attr in self.attrs.items():
            if schema is None or schema.has_attribute(name):
                onnx_attrs[name] = helper.make_attribute(name, attr)
        return onnx_attrs

    @property
    def name(self):
        return self._op.name

    @property
    def op(self):
        return self._op

    @property
    def op_type(self):
        """Return Op type."""
        return self._op.op_type

    @op_type.setter
    def op_type(self, val):
        """Set Op type."""
        self._op.op_type = val

    @property
    def domain(self):
        """Return Op type."""
        return self._op.domain

    @domain.setter
    def domain(self, val):
        """Set Op type."""
        self._op.domain = val

    @property
    def data_format(self):
        """Return data_format."""
        return self.attrs["data_format"]

    @data_format.setter
    def data_format(self, val):
        """Set data_format."""
        self.attrs["data_format"] = val

    def is_nhwc(self):
        """Return True if node is in NHWC format."""
        if self.op_type == "BatchNormalization":
            axis = self.attrs["axis"]
            return axis == -1 or axis == len(self.output_shapes[0]) - 1
        return self.data_format in ["NHWC", "channels_last"]

    def is_const(self):
        """Return True if node is a constant."""
        return self.op_type in ["variable", "Const"]
        # return self.op_type in ["Const", "ConstV2"]

    def is_graph_output(self):
        return self.op_type in ["return"]

    def is_graph_input(self):
        return self.op_type in ["input"]

    def is_graph_input_default_const(self):
        return self.is_const() and any(
            out.is_graph_input()
            for out in self.graph.FindOutputConsumers(self.output_tensor_names[0])
        )

    def __str__(self):
        return str(self._op)

    def __repr__(self):
        return "<onnx op type='%s' name=%s>" % (self.op_type, self._op.name)

    @property
    def summary(self):
        """Return node summary information."""
        lines = []
        lines.append("OP={}".format(self.op_type))
        lines.append("Name={}".format(self.name))

        g = self.graph
        if self.input_tensor_names:
            lines.append("Inputs:")
            for name in self.input_tensor_names:
                node = g.get_node_by_output(name)
                op = node.op_type if node else "N/A"
                lines.append(
                    "\t{}={}, {}, {}".format(
                        name, op, g.get_shape(name), g.get_dtype(name)
                    )
                )

        if self.output_tensor_names:
            for name in self.output_tensor_names:
                lines.append("Outpus:")
                lines.append(
                    "\t{}={}, {}".format(name, g.get_shape(name), g.get_dtype(name))
                )

        return "\n".join(lines)

    # If some Node is created as onnx_node, then we don't need convert it
    @property
    def skip_conversion(self):
        return self._skip_conversion

    @skip_conversion.setter
    def skip_conversion(self, val):
        self._skip_conversion = val

    @property
    def output_shapes(self):
        """Get output shapes."""
        self._GraphCheck()
        val = [self.graph.get_shape(n) for n in self._output]
        return val

    @property
    def output_dtypes(self):
        """Get output dtypes."""
        self._GraphCheck()
        val = [self.graph.get_dtype(n) for n in self._output]
        return val

    def get_tensor_value(self, as_list=True):
        """Get value for onnx tensor.
        Args:
            as_list: whether return numpy ndarray in list.
        Returns:
            If as_list=True, return the array as a (possibly nested) list.
            Otherwise, return data of type np.ndarray.

            If a tensor is a scalar having value 1,
                when as_list=False, return np.array(1), type is <class 'numpy.ndarray'>
                when as_list=True, return 1, type is <class 'int'>.
        """
        if not self.is_const():
            raise ValueError("get tensor value: {} must be Const".format(self.name))
        t = self.attrs.get("value", None)
        if t:
            t = numpy_helper.to_array(t)
        else:
            self._GraphCheck()
            t = self.graph.get_saved_tensor(self)
        if as_list is True and t is not None:
            t = t.tolist()  # t might be scalar after tolist()
        return t

    def ScalarTo1DTensor(self):
        """Get value for onnx tensor."""
        if not self.is_const():
            raise ValueError("get tensor value: {} must be Const".format(self.name))

        t = self.get_attr("value")
        if t:
            t = helper.get_attribute_value(t)
            if not t.dims:
                t.dims.extend([1])
        return t.dims

    def set_tensor_value(self, new_val):
        """Set new value for existing onnx tensor.
        Args:
            new_val: value of type numpy ndarray
        """
        if not self.is_const():
            raise ValueError("set tensor value: {} must be Const".format(self.name))
        t = self.attrs.get("value")
        if t is not None:
            t = helper.get_attribute_value(t)
            del t
        if self.op_type == "Const":
            tensor_name = t.name
        else:
            tensor_name = self.output_tensor_names[0]
        onnx_tensor = util.TensorProtoFromNumpy(new_val, tensor_name)
        self.attrs["value"] = onnx_tensor
        # track shapes in _output_shapes
        self._GraphCheck()
        self.graph.set_shape(onnx_tensor.name, onnx_tensor.dims)

    def get_body_graphs(self):
        self._GraphCheck()
        return self.graph.contained_graphs.get(self.name, None)

    def set_body_graph_as_attr(self, attr_name, graph):
        self._GraphCheck()
        if self.name not in self.graph.contained_graphs:
            self.graph.contained_graphs[self.name] = {}

        self.graph.contained_graphs[self.name].update({attr_name: graph})
        graph.parent_graph = self.graph

    def UpdateProto(self):
        """Update protobuf from internal structure."""
        nodes = list(self._op.input)
        for node in nodes:
            self._op.input.remove(node)
        self._op.input.extend(self.input_tensor_names)
        nodes = list(self._op.output)
        for node in nodes:
            self._op.output.remove(node)
        self._op.output.extend(self.output_tensor_names)

        # update attributes to proto
        del self._op.attribute[:]

        # check attribute of type GraphProto
        attr_graphs = self.get_body_graphs()
        if attr_graphs:
            for attr_name, sub_graph in attr_graphs.items():
                graph_proto = sub_graph.MakeGraph(
                    "graph for " + self.name + " " + attr_name
                )
                self.set_attr(attr_name, graph_proto)

        attr = list(self.attrs_onnx.values())
        if attr:
            self._op.attribute.extend(attr)

    def get_implicit_inputs(self, recursive=True):
        """Get implicit inputs if the node has attributes being GraphProto."""
        output_available_in_cur_graph = set()
        all_node_inputs = set()

        graphs = []
        body_graphs = self.get_body_graphs()
        if body_graphs:
            graphs.extend(body_graphs.values())

        while graphs:
            graph = graphs.pop()
            for n in graph.get_nodes():
                output_available_in_cur_graph |= set(n.output_tensor_names)
                for i in n.input_tensor_names:
                    all_node_inputs.add(i)

                if recursive:
                    b_graphs = n.get_body_graphs()
                    if b_graphs:
                        graphs.extend(b_graphs.values())

        outer_scope_node_input_ids = all_node_inputs - output_available_in_cur_graph
        return list(outer_scope_node_input_ids)

    def _GraphCheck(self):
        util.MakeSure(
            self.graph is not None, "Node %s not belonging any graph", self.name
        )
