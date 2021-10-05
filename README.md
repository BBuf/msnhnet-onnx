## msnhnet_onnx

**[简体中文](README.md) | [English](README_en.md)**

MsnhNet 相关的模型转换工具

### msnhnet_onnx

[![PyPI version](https://img.shields.io/pypi/v/msnhnet-onnx.svg)](https://pypi.python.org/pypi/msnhnet-onnx/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/msnhnet-onnx.svg)](https://pypi.python.org/pypi/msnhnet-onnx/)
[![PyPI license](https://img.shields.io/pypi/l/msnhnet-onnx.svg)](https://pypi.python.org/pypi/msnhnet-onnx/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Oneflow-Inc/oneflow_convert_tools/pulls)

#### 简介

msnhnet_onnx 转换工具支持将Pytorch/PaddlePaddle/TensorFlow2/OneFlow等框架构建的模型经过ONNX转换为MsnhNet可用的模型（`.mshnet`和`.bin`）

#### 环境依赖

##### 用户环境配置

```sh
python>=3.5
onnx>=1.8.0
onnx-simplifier>=0.3.3
onnxoptimizer>=0.2.5
onnxruntime>=1.6.0
pytorch>=1.7.0
paddlepaddle>=2.0.0
paddle2onnx>=0.6
tensorflow>=2.0.0
tf2onnx>=1.8.4
```

#### 安装

##### 安装方式1

```sh
pip install msnhnet_onnx
```

**安装方式2**

```
git clone https://github.com/BBuf/msnhnet-onnx
cd msnhnet_onnx
python3 setup.py install
```

#### 使用方法

请参考[使用示例](examples/README.md)

#### 相关文档

- [OneFlow2ONNX模型列表](docs/msnhnet2onnx/msnhnet2onnx_model_zoo.md)


### 项目进展


- 2021/10/5 初始化项目




