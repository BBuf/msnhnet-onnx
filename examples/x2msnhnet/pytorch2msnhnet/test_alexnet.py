import torchvision

from msnhnet_onnx.x2msnhnet.util import load_pytorch_module_and_check

alexnet = torchvision.models.alexnet(True)
alexnet.eval()

def test_alexnet():
    load_pytorch_module_and_check(
        alexnet,
        input_size=(1, 3, 227, 227),
        train_flag=False,
        msnhnet_weight_dir="/tmp/alexnet"
    )

test_alexnet()

