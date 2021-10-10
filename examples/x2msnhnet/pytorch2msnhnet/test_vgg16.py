import torchvision

from msnhnet_onnx.x2msnhnet.util import load_pytorch_module_and_check

vgg = torchvision.models.vgg16(True)
vgg.eval()

def test_vgg16():
    load_pytorch_module_and_check(
        vgg,
        input_size=(1, 3, 224, 224),
        train_flag=False,
        msnhnet_weight_dir="/tmp/vgg16"
    )

test_vgg16()
