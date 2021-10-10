import torchvision

from msnhnet_onnx.x2msnhnet.util import load_pytorch_module_and_check

resnet18 = torchvision.models.resnet18(True)
resnet18.eval()

def test_resnet18():
    load_pytorch_module_and_check(
        resnet18,
        input_size=(1, 3, 224, 224),
        train_flag=False,
        msnhnet_weight_dir="/tmp/resnet18"
    )

test_resnet18()
