
from __future__ import absolute_import
import setuptools

long_description = "msnhnet_onnx is a toolkit for converting trained model of Pytorch/PaddlePaddle/TensorFlow/OneFlow to ONNX and ONNX to MsnhNet.\n\n"
long_description += "Usage: msnhnet_onnx --model_dir src --save_file dist\n"
long_description += "GitHub: https://github.com/BBuf/msnhnet-onnx\n"
long_description += "Email: 1182563586@qq.com"

setuptools.setup(
    name="msnhnet_onnx",
    version="0.0.1",
    author="zhangxiaoyu",
    author_email="1182563586@qq.com",
    description="a toolkit for converting ONNX model to msnhnet.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/BBuf/msnhnet-onnx",
    packages=setuptools.find_packages(),
    install_requires=['six', 'protobuf'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0'
)
