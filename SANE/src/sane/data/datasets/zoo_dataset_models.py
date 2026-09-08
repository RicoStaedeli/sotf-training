from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as ResNetBase

import logging

from timm import create_model

from sane.utils.model_utils import get_activation_layer, initialize_weights
from .zoo_dataset_models_resnet_width import ResNet_width

class MLP(nn.Module):
    def __init__(
        self,
        i_dim=14,
        h_dim=[30, 15],
        o_dim=10,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
        use_bias=True,
    ):
        super().__init__()
        self.use_bias = use_bias
        # init module list
        self.module_list = nn.ModuleList()

        # get hidden layer's list
        # wrap h_dim in list of it's not already
        if not isinstance(h_dim, list):
            h_dim = [h_dim]

        # add i_dim to h_dim
        h_dim.insert(0, i_dim)

        # get if bias should be used or not
        for k in range(len(h_dim) - 1):
            # add linear layer
            self.module_list.append(
                nn.Linear(h_dim[k], h_dim[k + 1], bias=self.use_bias)
            )
            # add nonlinearity
            self.module_list.append(get_activation_layer(nlin))
            if dropout > 0:
                self.module_list.append(nn.Dropout(dropout))
        # init output layer
        self.module_list.append(nn.Linear(h_dim[-1], o_dim, bias=self.use_bias))
        # normalize outputs between 0 and 1
        # self.module_list.append(nn.Sigmoid())

        # initialize weights with se methods
        initialize_weights(self, init_type)

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            logging.debug(f"layer {layer}")
            logging.debug(f"input shape:: {x.shape}")
            x = layer(x)
            logging.debug(f"output shape:: {x.shape}")
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            activations.append(x)
        return x, activations


class CNN(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        # apply functional dropout, don't add to module list
        self.dropout = dropout
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(get_activation_layer(nlin))
        ## add linear layer 2
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        initialize_weights(self, init_type)

    def forward(self, x):
        # forward prop through module_list
        for idx, layer in enumerate(self.module_list):
            x = layer(x)
            if self.dropout > 0 and idx in [2, 5, 10]:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for idx, layer in enumerate(self.module_list):
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
            if self.dropout > 0 and idx in [2, 5, 10]:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x, activations


class CNN2(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(6, 9, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(9, 6, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 6, 20))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        initialize_weights(self, init_type)

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x)
        return x, activations


class CNN3(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 32x32 image size
        ## chn_in * 32 * 32
        ## compose layer 0
        self.module_list.append(nn.Conv2d(channels_in, 16, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 16 * 15 * 15
        ## compose layer 1
        self.module_list.append(nn.Conv2d(16, 32, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 32 * 7 * 7 // 32 * 6 * 6
        ## compose layer 2
        self.module_list.append(nn.Conv2d(32, 15, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 15 * 2 * 2
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(15 * 2 * 2, 20))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        initialize_weights(self, init_type)

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


class ResCNN(nn.Module):
    """
    extension of the CNN class.
    Added residual connections via max-pooling and 1d convs to the conv layers
    Added a function to load state-dicts from the cnns without res cons
    """

    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()

        if dropout > 0.0:
            raise NotImplementedError(
                "dropout is not yet impemented for the residual connections.|"
            )
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        # input shape bx1x28x28

        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 8, 12, 12]
        ## residual connection stack 1
        self.res1_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res1_conv = nn.Conv2d(
            in_channels=channels_in, out_channels=8, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 6, 4, 4]
        self.res2_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res2_conv = nn.Conv2d(
            in_channels=8, out_channels=6, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## output [15, 4, 3, 3]
        self.res3_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.res3_conv = nn.Conv2d(
            in_channels=6, out_channels=4, kernel_size=1, stride=1, padding=0
        )

        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, 10))

        ### initialize weights with se methods
        initialize_weights(self, init_type)

    def load_weights_from_cnn_checkpoint(self, check):
        """
        takes weights and biases from CNN architecture without residual connections
        assumes this particular data structure, will not fail gracefully otherwise
        """
        self.module_list[0].weight.data = check["module_list.0.weight"]
        self.module_list[0].bias.data = check["module_list.0.bias"]
        self.module_list[3].weight.data = check["module_list.3.weight"]
        self.module_list[3].bias.data = check["module_list.3.bias"]
        self.module_list[6].weight.data = check["module_list.6.weight"]
        self.module_list[6].bias.data = check["module_list.6.bias"]
        self.module_list[9].weight.data = check["module_list.9.weight"]
        self.module_list[9].bias.data = check["module_list.9.bias"]
        self.module_list[11].weight.data = check["module_list.11.weight"]
        self.module_list[11].bias.data = check["module_list.11.bias"]

    def forward(self, x):
        # layer 1
        x_ = x.clone()
        for idx in range(0, 3):
            x_ = self.module_list[idx](x_)
        x = self.res1_pool(x)
        x = self.res1_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 2
        x_ = x.clone()
        for idx in range(3, 6):
            x_ = self.module_list[idx](x_)
        x = self.res2_pool(x)
        x = self.res2_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 3
        x_ = x.clone()
        for idx in range(6, 8):
            x_ = self.module_list[idx](x_)
        x = self.res3_pool(x)
        x = self.res3_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # flatten
        # fc1
        # fc2
        for m in self.module_list[8:]:
            x = m(x)

        return x


class CNN_more_layers(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, 8, 5),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(8, 4, 5, bias=False),
            nn.Tanh(),
            nn.Conv2d(4, 6, 4, bias=False),
            nn.Tanh(),
            nn.Conv2d(6, 4, 3, bias=False),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(36, 18),
            nn.Tanh(),
            nn.Linear(18, 10),
        )

        initialize_weights(self, init_type)

    def forward(self, x):
        return self.layers(x)


class CNN_residual(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 6, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(6, 4, 2, bias=False)
        self.act3 = nn.Tanh()

        self.identity = nn.Conv2d(8, 4, 1, bias=False)

        self.flatten = nn.Flatten()

        self.fc4 = nn.Linear(36, 20, bias=False)
        self.act4 = nn.Tanh()

        self.fc5 = nn.Linear(20, 10)

        initialize_weights(self, init_type)

    def forward(self, x):
        x = self.conv1(x)
        logging.debug(x.shape)

        x = self.act1(self.pool1(x))
        logging.debug(x.shape)

        x_identity = self.identity(x)
        logging.debug("x_identity", x_identity.shape)

        x = self.act2(self.pool2(self.conv2(x)))
        logging.debug(x.shape)

        x = self.act3(self.conv3(x))
        logging.debug(x.shape)

        x = x + x_identity[:, :, ::4, ::4]

        logging.debug("skip connection applied")

        x = self.flatten(x)

        x = self.act4(self.fc4(x))
        logging.debug(x.shape)

        x = self.fc5(x)
        logging.debug(x.shape)

        return x


class CNN_more_layers_residual(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 4, 5, bias=False)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(4, 6, 4, bias=False)
        self.act3 = nn.Tanh()

        self.conv4 = nn.Conv2d(6, 4, 3, bias=False)
        self.act4 = nn.Tanh()

        self.flatten = nn.Flatten()

        self.fc5 = nn.Linear(36, 18)
        self.act5 = nn.Tanh()

        self.fc6 = nn.Linear(18, 10)

        initialize_weights(self, init_type)

    def forward(self, x):
        x = self.conv1(x)
        logging.debug(x.shape)

        x = self.act1(self.pool1(x))
        logging.debug(x.shape)

        x = self.act2(self.conv2(x))
        logging.debug(x.shape)

        x_identity = x.clone()

        x = self.act3(self.conv3(x))
        logging.debug(x.shape)

        x = self.act4(self.conv4(x))
        logging.debug(x.shape)

        x = x + x_identity[:, :, 1:-1, 1:-1][:, :, ::2, ::2]

        logging.debug("skip connection applied")

        x = self.flatten(x)

        x = self.act5(self.fc5(x))
        logging.debug(x.shape)

        x = self.fc6(x)
        logging.debug(x.shape)

        return x


class ResNet(ResNetBase):
    """
    Wrapper for pytroch ResNet class to get access to forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_activations(self, x):
        """forward pass and return activations"""
        # See note [TorchScript super()]
        activations = []

        # input block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        activations.append(x.clone())

        # residual blocks
        x = self.layer1(x)
        activations.append(x.clone())
        x = self.layer2(x)
        activations.append(x.clone())
        x = self.layer3(x)
        activations.append(x.clone())
        x = self.layer4(x)
        activations.append(x.clone())

        # output block
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, activations


class ResNetBase(ResNet):
    """
    ResNet base class, defaults to ResNet 18, implements init and weight init
    """

    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    ):
        # call init from parent class
        super().__init__(block=block, layers=layers, num_classes=out_dim)
        # adpat first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            initialize_weights(self, init_type)


class ResNet18(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


# class ResNet18Width(ResNet):
#     def __init__(
#         self,
#         channels_in=3,
#         out_dim=10,
#         nlin="relu",
#         dropout=0.0,
#         init_type="kaiming_uniform",
#         width=64,
#     ):
#         nn.Module.__init__(self)
#         norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#         self.inplanes = width
#         self.dilation = 1
#         self.groups = 1
#         self.base_width = 64  # required by BasicBlock

#         self.conv1 = nn.Conv2d(channels_in, width, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = norm_layer(width)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.Identity()

#         self.layer1 = self._make_layer(BasicBlock, width, 2)
#         self.layer2 = self._make_layer(BasicBlock, width * 2, 2, stride=2)
#         self.layer3 = self._make_layer(BasicBlock, width * 4, 2, stride=2)
#         self.layer4 = self._make_layer(BasicBlock, width * 8, 2, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(width * 8, out_dim)

#         if init_type is not None:
#             initialize_weights(self, init_type)

class ResNetBase_width(ResNet_width):
    """
    ResNet base class, defaults to ResNet 18, implements init and weight init
    """

    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        width=64,
    ):
        # call init from parent class
        super().__init__(block=block, layers=layers, num_classes=out_dim, width=width)
        # adpat first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            width,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m

class ResNet18Width(ResNetBase_width):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        width=64,
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
            width=width,
        )


class ResNet34(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[3, 4, 6, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet50(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 6, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet50Width(ResNet):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",
        dropout=0.0,
        init_type="kaiming_uniform",
        width=64,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
    ):
        nn.Module.__init__(self)
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = width
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(
            channels_in, width, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, width * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8 * block.expansion, out_dim)

        if init_type is not None:
            initialize_weights(self, init_type)


class ResNet101(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet152(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 8, 36, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=Bottleneck,
            layers=layers,
        )


class MiniAlexNet(nn.Module):
    def __init__(self, channels_in=3, num_classes=10, init_type="kaiming_uniform"):
        super(MiniAlexNet, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(
            channels_in, 96, kernel_size=3, stride=1
        )  # Use padding to keep size 32x32
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=3
        )  # Reduce size from 32x32 to 10x10
        self.batchnorm1 = nn.BatchNorm2d(96)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1)  # Keep size 10x10
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Reduce size from 10x10 to 5x5
        self.batchnorm2 = nn.BatchNorm2d(256)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 384)  # Adjusted for 5x5 feature map size
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

        if init_type is not None:
            initialize_weights(self, init_type)

    def forward(self, x):
        # Apply Convolutional Layers
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.maxpool2(x)

        x = x.view(-1, 256 * 4 * 4)

        # Apply Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class _VPLAttention(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.0))


class _VPLFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.0),
        )


class _VPLPreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn


class _VPLTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, inner_dim: int, mlp_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                _VPLPreNorm(dim, _VPLAttention(dim, inner_dim)),
                _VPLPreNorm(dim, _VPLFeedForward(dim, mlp_dim)),
            ])
            for _ in range(depth)
        ])


class ViTPytorchLegacy(nn.Module):
    """Mirrors the vit-pytorch ViT key layout used in pre-timm zoo checkpoints.

    Parameters are inferred from the state dict — do not construct directly,
    use :func:`vit_pytorch_legacy_from_state_dict` instead.
    """

    def __init__(self, patch_dim: int, dim: int, depth: int, inner_dim: int, mlp_dim: int,
                 num_patches: int, num_classes: int):
        super().__init__()
        # Index 0 is Rearrange (no params); Identity is a stand-in so indices 1-3 are preserved.
        self.to_patch_embedding = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.transformer = _VPLTransformer(dim, depth, inner_dim, mlp_dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))


def vit_pytorch_legacy_from_state_dict(state_dict: dict) -> "ViTPytorchLegacy":
    """Build a :class:`ViTPytorchLegacy` whose architecture matches *state_dict*."""
    proj_w = state_dict["to_patch_embedding.2.weight"]
    dim, patch_dim = proj_w.shape
    depth = max(int(k.split(".")[2]) for k in state_dict if k.startswith("transformer.layers.")) + 1
    num_classes = state_dict["mlp_head.1.weight"].shape[0]
    mlp_dim = state_dict["transformer.layers.0.1.fn.net.0.weight"].shape[0]
    inner_dim = state_dict["transformer.layers.0.0.fn.to_qkv.weight"].shape[0] // 3
    num_patches = state_dict["pos_embedding"].shape[1] - 1
    return ViTPytorchLegacy(patch_dim, dim, depth, inner_dim, mlp_dim, num_patches, num_classes)


class ViTSmallPatch16(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 init_type="kaiming_uniform",
                 fc_mlp=False,
                 hidden_dim=256,
                 embedding_dim=384,
                 depth=12,
                 num_heads=6,
                 dropout=0.,
                 attn_dropout=0.,
                 img_size=32,
                 patch_size=4,
                 mixup=None,
                 cutmix=None,
                 random_erase=None,
                 pretrained_model_path=None):
        super(ViTSmallPatch16, self).__init__()
        self.vit = create_model("vit_small_patch16_224",
                                pretrained=False,
                                num_classes=num_classes,
                                img_size=img_size,
                                patch_size=patch_size,
                                embed_dim=embedding_dim,
                                depth=depth,
                                num_heads=num_heads,
                                proj_drop_rate=dropout,
                                attn_drop_rate=attn_dropout)

        if pretrained_model_path is not None:
            self.load_partial_state_dict(pretrained_model_path)
        else:
            self.initialize_weights(init_type)

        if fc_mlp:
            self.vit.head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.vit.head = nn.Linear(embedding_dim, num_classes)

        if init_type is not None:
            self.initialize_head(init_type)

        self.mixup = mixup
        self.cutmix = cutmix
        self.random_erase = random_erase

    def load_partial_state_dict(self, pretrained_model_path):
        state_dict = torch.load(pretrained_model_path)
        state_dict.pop('vit.head.weight', None)
        state_dict.pop('vit.head.bias', None)
        self.load_state_dict(state_dict, strict=False)

    def initialize_weights(self, init_type):
        for m in self.modules():
            m = self.init_single(init_type, m)

    def initialize_head(self, init_type):
        m = self.vit.head
        m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            try:
                m.bias.data.fill_(0.01)
            except Exception:
                pass
        return m

    def reset_classifier(self, num_classes=100, fc_mlp=False, hidden_dim=256):
        if fc_mlp:
            self.vit.head = nn.Sequential(
                nn.Linear(384, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.vit.head = nn.Linear(384, num_classes)

    def forward(self, x, y=None):
        if isinstance(x, list):
            x = torch.cat(x, dim=0).to("cuda" if torch.cuda.is_available() else "cpu")
        if self.training and y is not None:
            if self.mixup is not None:
                x, y = self.mixup(x, y)
            if self.random_erase is not None:
                x = self.random_erase(x)
            x = self.vit(x)
            return x, y
        else:
            x = self.vit(x)
            return x


def get_model(config:Dict) -> nn.Module:
    model_type = config["model::type"]
    match model_type:
        case "MLP":
            i_dim = config["model::i_dim"]
            h_dim = config["model::h_dim"]
            o_dim = config["model::o_dim"]
            nlin = config["model::nlin"]
            dropout = config["model::dropout"]
            init_type = config["model::init_type"]
            use_bias = config["model::use_bias"]

            return MLP(i_dim, h_dim, o_dim, nlin, dropout, init_type, use_bias)

        case "CNN":
            channels_in=config["model::channels_in"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return CNN(channels_in, nlin, dropout, init_type)

        case "CNN2":
            channels_in=config["model::channels_in"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return CNN2(channels_in, nlin, dropout, init_type)

        case "CNN3":
            channels_in=config["model::channels_in"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return CNN3(channels_in, nlin, dropout, init_type)

        case "ResCNN":
            channels_in=config["model::channels_in"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResCNN(channels_in, nlin, dropout, init_type)

        case "CNN_more_layers":
            init_type=config["model::init_type"]
            channels_in=config["model::channels_in"]

            return CNN_more_layers(init_type, channels_in)

        case "CNN_residual":
            init_type=config["model::init_type"]
            channels_in=config["model::channels_in"]

            return CNN_residual(init_type, channels_in)

        case "CNN_more_layers_residual":
            init_type=config["model::init_type"]
            channels_in=config["model::channels_in"]

            return CNN_more_layers_residual(init_type, channels_in)

        case "Resnet18":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet18(channels_in, out_dim, nlin, dropout, init_type)

        case "Resnet18_width":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]
            width=config["model::width"]

            return ResNet18Width(channels_in, out_dim, nlin, dropout, init_type, width=width)

        case "Resnet34":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet34(channels_in, out_dim, nlin, dropout, init_type)

        case "Resnet50":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet50(channels_in, out_dim, nlin, dropout, init_type)

        case "Resnet50_width" | "ResNet50_width" | "Resnet50Width" | "ResNet50Width":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]
            width=config["model::width"]

            return ResNet50Width(
                channels_in, out_dim, nlin, dropout, init_type, width
            )

        case "Resnet101":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet101(channels_in, out_dim, nlin, dropout, init_type)

        case "Resnet152":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet152(channels_in, out_dim, nlin, dropout, init_type)

        case "MiniAlexNet":
            channels_in=config["model::channels_in"]
            num_classes=config["model::o_dim"]
            init_type=config["model::init_type"]

            return MiniAlexNet(channels_in, num_classes, init_type)

        case "efficientnet_v2_s":
            num_classes=config["model::o_dim"]
            dropout=config["model::dropout"]

            return torchvision.models.efficientnet_v2_s(num_classes, dropout)

        case "efficientnet_v2_m":
            num_classes=config["model::o_dim"]
            dropout=config["model::dropout"]

            return torchvision.models.efficientnet_v2_m(num_classes, dropout)

        case "densenet121":
            num_classes=config["model::o_dim"]

            return torchvision.models.densenet121(num_classes)

        case "densenet161":
            num_classes=config["model::o_dim"]

            return torchvision.models.densenet161(num_classes)

        case "vit_b_16":
            num_classes=config["model::o_dim"]
            dropout=config["model::dropout"]

            return torchvision.models.vit_b_16(num_classes, dropout)

        case "ViT" | "ViTSmallPatch16":
            num_classes=config["model::o_dim"]
            init_type=config.get("model::init_type", None)
            fc_mlp=config.get("model::fc_mlp", False)
            hidden_dim=config.get("model::hidden_dim", 256)
            embedding_dim=config.get("model::embedding_dim", 384)
            depth=config.get("model::depth", 12)
            num_heads=config.get("model::heads", 6)
            dropout=config.get("model::dropout", 0.)
            attn_dropout=config.get("model::attn_dropout", 0.)
            img_size=config.get("model::image_size", config.get("model::img_size", 32))
            patch_size=config.get("model::patch_size", 4)

            return ViTSmallPatch16(
                num_classes=num_classes,
                init_type=init_type,
                fc_mlp=fc_mlp,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                depth=depth,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                img_size=img_size,
                patch_size=patch_size,
            )

        case _:
            raise NotImplementedError(f"error: unknown model type '{model_type}'")
