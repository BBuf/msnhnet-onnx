import torchvision

from msnhnet_onnx.x2msnhnet.util import load_pytorch_module_and_check

mobilenetv2 = torchvision.models.mobilenet_v2(True)
mobilenetv2.eval()

def test_mobilenet_v2():
    load_pytorch_module_and_check(
        mobilenetv2,
        input_size=(1, 3, 224, 224),
        train_flag=False,
        msnhnet_weight_dir="/tmp/mobilenetv2"
    )

test_mobilenet_v2()
