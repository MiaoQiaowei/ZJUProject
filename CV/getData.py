import torchvision
from torchvision import transforms



def get_train_test_set():
    trainset = torchvision.datasets.MNIST( 
    root="./",
    train=True, 
    transform=transforms.ToTensor(),
    download=False
)

    testset = torchvision.datasets.MNIST(
        root="./",
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )
    return trainset, testset