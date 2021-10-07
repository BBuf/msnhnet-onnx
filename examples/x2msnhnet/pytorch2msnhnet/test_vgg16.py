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
import torchvision

from msnhnet_onnx.x2msnhnet.util import load_pytorch_module_and_check


def test_vgg16():
    load_pytorch_module_and_check(
        torchvision.models.vgg16,
        input_size=(1, 3, 224, 224),
        train_flag=False,
        msnhnet_weight_dir="/tmp/vgg16"
    )

test_vgg16()
