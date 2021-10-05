import torchvision

from msnhnet_onnx.x2msnhnet.util import load_pytorch_module_and_check


def test_alexnet():
    load_pytorch_module_and_check(
        torchvision.models.alexnet,
        input_size=(1, 3, 224, 224),
        train_flag=False,
        msnhnet_weight_dir="/tmp/msnhnet"
    )

test_alexnet()

