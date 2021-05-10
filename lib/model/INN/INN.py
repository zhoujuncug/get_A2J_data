import torch
import torch.nn as nn
import numpy as np



CONFIG_MAP = {
    "cinn_alexnet_aae_conv5":
        {"Transformer": {
              "activation": "none",
              "conditioning_option": "none",
              "hidden_depth": 2,
              "in_channels": 128,
              "mid_channels": 1024,
              "n_flows": 20,
              "conditioning_in_channels": 256,
              "conditioning_spatial_size": 13,
              "embedder_down": 2,
            }
        },
    "cinn_alexnet_aae_fc6":
        {"Transformer": {
                      "activation": "none",
                      "conditioning_option": "none",
                      "hidden_depth": 2,
                      "in_channels": 128,
                      "mid_channels": 1024,
                      "n_flows": 20,
                      "conditioning_in_channels": 4096,
                      "conditioning_spatial_size": 1,
                      "embedder_down": 3,
                    }
                },
    "cinn_alexnet_aae_fc7":
        {"Transformer": {
                      "activation": "none",
                      "conditioning_option": "none",
                      "hidden_depth": 2,
                      "in_channels": 128,
                      "mid_channels": 1024,
                      "n_flows": 20,
                      "conditioning_in_channels": 4096,
                      "conditioning_spatial_size": 1,
                      "embedder_down": 3,
                    }
                },
    "cinn_alexnet_aae_fc8":
        {"Transformer": {
                      "activation": "none",
                      "conditioning_option": "none",
                      "hidden_depth": 2,
                      "in_channels": 128,
                      "mid_channels": 1024,
                      "n_flows": 20,
                      "conditioning_in_channels": 1000,
                      "conditioning_spatial_size": 1,
                      "embedder_down": 3,
                    }
                },
    "cinn_alexnet_aae_softmax":
        {"Transformer": {
            "activation": "none",
            "conditioning_option": "none",
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 1000,
            "conditioning_spatial_size": 1,
            "embedder_down": 3,
            }
        },
    "cinn_stylizedresnet_avgpool":
        {"Transformer": {
            "activation": "none",
            "conditioning_option": "none",
            "hidden_depth": 2,
            "in_channels": 268,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 1,
            "embedder_down": 3,
            }
        },
    "cinn_resnet_avgpool":
        {"Transformer": {
            "activation": "none",
            "conditioning_option": "none",
            "hidden_depth": 2,
            "in_channels": 268,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 1,
            "embedder_down": 3,
            }
        },
    "resnet101_animalfaces_shared":
        {"Model": {
            "n_classes": 149,
            "type": "resnet101"
            }
        },

    "resnet101_animalfaces_10":
        {"Model": {
                "n_classes": 10,
                "type": "resnet101"
                }
        },
    "cinn_resnet_animalfaces10_ae_maxpool":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 64,
            "conditioning_spatial_size": 56,
            "embedder_down": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_input":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 3,
            "conditioning_spatial_size": 224,
            "embedder_down": 5,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer1":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 256,
            "conditioning_spatial_size": 56,
            "embedder_down": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer2":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 512,
            "conditioning_spatial_size": 28,
            "embedder_down": 3,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer3":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 1024,
            "conditioning_spatial_size": 14,
            "embedder_down": 2,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer4":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 7,
            "embedder_down": 1,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_avgpool":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 1,
            "conditioning_depth": 6,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_fc":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 10,
            "conditioning_spatial_size": 1,
            "conditioning_depth": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_softmax":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 10,
            "conditioning_spatial_size": 1,
            "conditioning_depth": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
}

URL_MAP = {
    "cinn_alexnet_aae_conv5": "https://heibox.uni-heidelberg.de/f/62b0e29d8c544f51b79c/?dl=1",
    "cinn_alexnet_aae_fc6": "https://heibox.uni-heidelberg.de/f/5d07dc071dd1450eb0a5/?dl=1",
    "cinn_alexnet_aae_fc7": "https://heibox.uni-heidelberg.de/f/050d4d76f3cf4eeeb9b6/?dl=1",
    "cinn_alexnet_aae_fc8": "https://heibox.uni-heidelberg.de/f/cb9c93497aae4e97890c/?dl=1",
    "cinn_alexnet_aae_softmax": "https://heibox.uni-heidelberg.de/f/5a30088c51b44cc58bbe/?dl=1",
    "cinn_stylizedresnet_avgpool": "https://heibox.uni-heidelberg.de/f/7a54929dee4544248020/?dl=1",
    "cinn_resnet_avgpool": "https://heibox.uni-heidelberg.de/f/d31717bb225b4692bbb3/?dl=1",
    "resnet101_animalfaces_shared": "https://heibox.uni-heidelberg.de/f/a2c36d628f574ed8aa68/?dl=1",
    "resnet101_animalfaces_10": "https://heibox.uni-heidelberg.de/f/314926cb0d754cd9bb02/?dl=1",
    "cinn_resnet_animalfaces10_ae_maxpool": "https://heibox.uni-heidelberg.de/f/30dc2640dfd54b339f93/?dl=1",
    "cinn_resnet_animalfaces10_ae_input": "https://heibox.uni-heidelberg.de/f/abe885bdaf1c46139020/?dl=1",
    "cinn_resnet_animalfaces10_ae_layer1": "https://heibox.uni-heidelberg.de/f/00687f7bc1d6409aa680/?dl=1",
    "cinn_resnet_animalfaces10_ae_layer2": "https://heibox.uni-heidelberg.de/f/51795cdc045246c49cdc/?dl=1",
    "cinn_resnet_animalfaces10_ae_layer3": "https://heibox.uni-heidelberg.de/f/860cfa3327f3426db1f6/?dl=1",
    "cinn_resnet_animalfaces10_ae_layer4": "https://heibox.uni-heidelberg.de/f/1a2058efb9564cb7bcf7/?dl=1",
    "cinn_resnet_animalfaces10_ae_avgpool": "https://heibox.uni-heidelberg.de/f/859c201776ee4b5f8b15/?dl=1",
    "cinn_resnet_animalfaces10_ae_fc": "https://heibox.uni-heidelberg.de/f/b0a485885ab44e4daa44/?dl=1",
    "cinn_resnet_animalfaces10_ae_softmax": "https://heibox.uni-heidelberg.de/f/19c56eff387c40f3bd44/?dl=1",
}


CKPT_MAP = {
    "cinn_alexnet_aae_conv5": "invariances/pretrained_models/cinns/alexnet/conv5.ckpt",
    "cinn_alexnet_aae_fc6": "invariances/pretrained_models/cinns/alexnet/fc6.ckpt",
    "cinn_alexnet_aae_fc7": "invariances/pretrained_models/cinns/alexnet/fc7.ckpt",
    "cinn_alexnet_aae_fc8": "invariances/pretrained_models/cinns/alexnet/fc8.ckpt",
    "cinn_alexnet_aae_softmax": "invariances/pretrained_models/cinns/alexnet/softmax.ckpt",
    "cinn_stylizedresnet_avgpool": "invariances/pretrained_models/cinns/stylized_resnet50/avgpool.ckpt",
    "cinn_resnet_avgpool": "invariances/pretrained_models/cinns/resnet50/avgpool.ckpt",
    "resnet101_animalfaces_shared": "invariances/pretrained_models/classifiers/resnet101/animalfaces149_modelub_16908.ckpt",
    "resnet101_animalfaces_10": "invariances/pretrained_models/classifiers/resnet101/animalfaces10_modelub_6118.ckpt",
    "cinn_resnet_animalfaces10_ae_maxpool": "invariances/pretrained_models/cinns/maxpool_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_input": "invariances/pretrained_models/cinns/input_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_layer1": "invariances/pretrained_models/cinns/layer1_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_layer2": "invariances/pretrained_models/cinns/layer2_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_layer3": "invariances/pretrained_models/cinns/layer3_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_layer4": "invariances/pretrained_models/cinns/layer4_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_avgpool": "invariances/pretrained_models/cinns/avgpool_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_fc": "invariances/pretrained_models/cinns/fc_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_softmax": "invariances/pretrained_models/cinns/softmax_model-7000.ckpt",
}


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class FeatureLayer(nn.Module):
    def __init__(self, scale, in_channels=None, norm='AN', width_multiplier=1):
        # scale = 0,
        # in_channels = 256
        # norm = 'an'
        super().__init__()

        norm_options = {
            "in": nn.InstanceNorm2d,
            "bn": nn.BatchNorm2d,
            "an": ActNorm}

        self.scale = scale  # 0   # 1
        self.norm = norm_options[norm.lower()]  # ActNorm
        self.wm = width_multiplier  # 1
        if in_channels is None:
            self.in_channels = int(self.wm * 64 * min(2 ** (self.scale - 1), 16))   #    # 64
        else:
            self.in_channels = in_channels  # 256
        self.out_channels = int(self.wm * 64 * min(2 ** self.scale, 16))  # 64   # 128
        self.build()
        # self.sub_layers:
        # Conv2d(256, 64, 4, 2, 1, b=False)
        # ActNorm()
        # LeakyReLU

        # self.sub_layers:
        # Conv2d(64, 128, 4, 2, 1, b=False)
        # ActNorm()
        # LeakyReLU(0.2)

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        Norm = ActNorm
        Activate = lambda: nn.LeakyReLU(0.2)
        self.sub_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            Norm(num_features=self.out_channels),
            Activate()])


class DenseEncoderLayer(nn.Module):
    def __init__(self, scale, spatial_size, out_size, in_channels=None,
                 width_multiplier=1):
        # scale = 2
        # spatial_size = 3
        # out_size = 128
        # in_channels = None
        # width_multiplier = 1
        super().__init__()
        self.scale = scale  # 2
        self.wm = width_multiplier  # 1
        self.in_channels = int(self.wm * 64 * min(2 ** (self.scale - 1), 16))  # 128
        if in_channels is not None:
            print('Warning: Ignoring `scale` parameter in DenseEncoderLayer due to given number of input channels.')
            self.in_channels = in_channels
        self.out_channels = out_size  # 128
        self.kernel_size = spatial_size  # 3
        self.build()
        # self.sub_layers:
        # Conv2d(128, 128, 3, 1, )

    def forward(self, input):
        x = input
        for layer in self.sub_layers:
            x = layer(x)
        return x

    def build(self):
        self.sub_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=0,
                bias=True)])


class DenseEmbedder(nn.Module):
    """Basically an MLP. Maps vector-like features to some other vector of given dimenionality"""

    def __init__(self, in_dim, up_dim, depth=4, given_dims=None):
        super().__init__()
        self.net = nn.ModuleList()
        if given_dims is not None:
            assert given_dims[0] == in_dim
            assert given_dims[-1] == up_dim
            dims = given_dims
        else:
            dims = np.linspace(in_dim, up_dim, depth).astype(int)
        for l in range(len(dims) - 2):
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))
            self.net.append(ActNorm(dims[l + 1]))
            self.net.append(nn.LeakyReLU(0.2))

        self.net.append(nn.Conv2d(dims[-2], dims[-1], 1))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x.squeeze(-1).squeeze(-1)


class Embedder(nn.Module):
    """Embeds a 4-dim tensor onto dense latent code."""

    def __init__(self, in_spatial_size, in_channels, emb_dim, n_down=4):
        # in_spatial_size = 13
        # in_channels = 256
        # emb_dim = 128
        # n_down = 2
        super().__init__()
        self.feature_layers = nn.ModuleList()
        norm = 'an'  # hard coded
        bottleneck_size = in_spatial_size // 2 ** n_down  # 3
        self.feature_layers.append(FeatureLayer(0,
                                                in_channels=in_channels,  # 256
                                                norm=norm))  # 'an

        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale,  # 1
                                                    norm=norm))  # 'an'
        # self.feature_layers:
        #  Conv2d(256, 64, 4, 2, 1, b=False)
        #  ActNorm()
        #  LeakyReLU(0.2)

        #  Conv2d(64, 128, 4, 2, 1, b=False)
        #  ActNorm()
        #  LeakyReLU(0.2)

        self.dense_encode = DenseEncoderLayer(n_down,  # 2
                                              bottleneck_size,  # 3
                                              emb_dim)  # 128
        # self.dense_encode:
        # Conv2d(128, 128, 3, 1, )

        if n_down == 1:
            print(" Warning: Embedder for ConditionalTransformer has only one down-sampling step. You might want to "
                  "increase its capacity.")

    def forward(self, input):
        h = input
        # h: (16, 256, 13, 13)
        for layer in self.feature_layers:
            h = layer(h)
        # h: (16, 128, 3, 3)
        h = self.dense_encode(h)
        # h: (16, 128, 1, 1)
        return h.squeeze(-1).squeeze(-1)


class IgnoreLeakyRelu(nn.Module):
    """performs identity op."""

    def __init__(self, alpha=0.9):
        super().__init__()

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        h = input
        return h, 0.0

    def reverse(self, input):
        h = input
        return h


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False, out_dim=None):
        # dim = 192
        # depth = 2
        # hidden_dim = 1024
        # use_tanh = True,
        # use_bn = False
        # out_dim = 64
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))  # 192 --> 1024
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)
        # self.main: Linear(192, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 64) Tanh

    def forward(self, x):
        for m in self.main:
            x = m(x)
        return x #  self.main(x)


class ConditionalDoubleVectorCouplingBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, depth=2):
        # in_channels = 128
        # cond_channels = 128
        # hidden_dim = 1024
        # hidden_depth = 2
        super(ConditionalDoubleVectorCouplingBlock, self).__init__()
        self.s = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels // 2 + cond_channels,  # 192
                                   depth=depth,  # 2
                                   hidden_dim=hidden_dim,  # 1024
                                   use_tanh=True, use_bn=False,
                                   out_dim=in_channels // 2)  # 64
            for _ in range(2)])
        # self.s
        # BasicFullyConnectedNet:
        #            Linear(192, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 64) Tanh
        #
        # BasicFullyConnectedNet:
        #            Linear(192, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 64) Tanh

        self.t = nn.ModuleList([
            BasicFullyConnectedNet(dim=in_channels // 2 + cond_channels, depth=depth,
                                   hidden_dim=hidden_dim, use_tanh=False, use_bn=False,
                                   out_dim=in_channels // 2) for _ in range(2)])
        # self.t
        # BasicFullyConnectedNet:
        #            Linear(192, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 64)
        #
        # BasicFullyConnectedNet:
        #            Linear(192, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 1024) LeakyReLU
        #            Linear(1024, 64)

    def forward(self, x, xc, reverse=False):
        if len(x.shape) == 2:
            x = x[:, :, None, None]
        if len(xc.shape) == 2:
            xc = xc[:, :, None, None]
        if len(xc.shape) == 6:
            xc = xc[:, :, :, :, 0, 0]
        assert len(x.shape) == 4
        assert len(xc.shape) == 4
        # x: (16, 128, 1, 1)
        # xc: (16, 128, 1, 1)
        x = x.squeeze(-1).squeeze(-1)
        xc = xc.squeeze(-1).squeeze(-1)
        if not reverse:
            logdet = 0
            for i in range(len(self.s)):
                idx_apply, idx_keep = 0, 1
                if i % 2 != 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                x = torch.chunk(x, 2, dim=1)
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)
                scale = self.s[i](conditioner_input)
                x_ = x[idx_keep] * scale.exp() + self.t[i](conditioner_input)
                x = torch.cat((x[idx_apply], x_), dim=1)
                logdet_ = torch.sum(scale, dim=1)
                logdet = logdet + logdet_  # (16,)
            
            return x[:, :, None, None], logdet
        else:
            idx_apply, idx_keep = 0, 1
            for i in reversed(range(len(self.s))):
                # x: (16, 128)
                if i % 2 == 0:
                    x = torch.cat(torch.chunk(x, 2, dim=1)[::-1], dim=1)
                # x: (16, 128)
                x = torch.chunk(x, 2, dim=1)
                # x: [(16, 64), (16, 64)]
                conditioner_input = torch.cat((x[idx_apply], xc), dim=1)  # x: (16, 192)
                x_ = (x[idx_keep] - self.t[i](conditioner_input)) * self.s[i](conditioner_input).neg().exp()
                x = torch.cat((x[idx_apply], x_), dim=1)
            return x[:, :, None, None]


class Shuffle(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Shuffle, self).__init__()
        self.in_channels = in_channels
        idx = torch.randperm(in_channels)
        self.register_buffer('forward_shuffle_idx', nn.Parameter(idx, requires_grad=False))
        self.register_buffer('backward_shuffle_idx', nn.Parameter(torch.argsort(idx), requires_grad=False))

    def forward(self, x, reverse=False, conditioning=None):
        if not reverse:
            return x[:, self.forward_shuffle_idx, ...], 0
        else:
            return x[:, self.backward_shuffle_idx, ...]


class ConditionalFlatDoubleCouplingFlowBlock(nn.Module):
    def __init__(self, in_channels, cond_channels, hidden_dim, hidden_depth, activation="lrelu"):
        # in_channels = 128
        # cond_channels = 128
        # hidden_dim = 1024
        # hidden_depth = 2
        # activation = None
        super().__init__()
        __possible_activations = {
                                  "none": IgnoreLeakyRelu
                                  }
        self.norm_layer = ActNorm(in_channels, logdet=True)
        self.coupling = ConditionalDoubleVectorCouplingBlock(in_channels,  # 128
                                                             cond_channels,  # 128
                                                             hidden_dim,  # 1024
                                                             hidden_depth)  # 2
        self.activation = __possible_activations[activation]()
        self.shuffle = Shuffle(in_channels)

    def forward(self, x, xcond, reverse=False):
        if not reverse:
            h = x  # (16, 128, 1, 1)
            logdet = 0.0
            h, ld = self.norm_layer(h)
            # ld: (16,)
            logdet += ld
            h, ld = self.activation(h)  # Nothing happened
            # ld = 0.0
            logdet += ld
            if len(h.shape) == 2:
                h = h[:, :, None, None]
            h, ld = self.coupling(h, xcond)  # (16, 128, 1, 1)
            # ld: (16,)
            logdet += ld
            h, ld = self.shuffle(h)  # what if no shuffle
            # ld = 0
            logdet += ld
            return h, logdet
        else:
            h = x
            h = self.shuffle(h, reverse=True)
            h = self.coupling(h, xcond, reverse=True)
            h = self.activation(h, reverse=True)
            h = self.norm_layer(h, reverse=True)
            return h

    def reverse(self, out, xcond):
        return self.forward(out, xcond, reverse=True)


class ConditionalFlow(nn.Module):
    """Flat version. Feeds an embedding into the flow in every block"""

    def __init__(self, in_channels, embedding_dim, hidden_dim, hidden_depth,
                 n_flows, conditioning_option="none", activation='lrelu'):
        # in_channels = 128
        # embedding_dim = 128
        # hidden_dim = 1024
        # hidden_depth = 2
        # n_flows = 20
        # conditioning_option = None
        # activation = None
        super().__init__()
        self.in_channels = in_channels  # 128
        self.cond_channels = embedding_dim  # 128
        self.mid_channels = hidden_dim  # 1024
        self.num_blocks = hidden_depth  # 2
        self.n_flows = n_flows  # 20
        self.conditioning_option = conditioning_option  # None

        self.sub_layers = nn.ModuleList()
        if self.conditioning_option.lower() != "none":
            self.conditioning_layers = nn.ModuleList()
        for flow in range(self.n_flows):  # 20
            self.sub_layers.append(ConditionalFlatDoubleCouplingFlowBlock(
                self.in_channels,  # 128
                self.cond_channels,  # 128
                self.mid_channels,  # 1024
                self.num_blocks,  # 2
                activation=activation)  # None
            )
            if self.conditioning_option.lower() != "none":
                self.conditioning_layers.append(nn.Conv2d(self.cond_channels, self.cond_channels, 1))

    def forward(self, x, embedding, reverse=False):
        hconds = list()
        hcond = embedding[:, :, None, None]
        self.last_outs = []
        self.last_logdets = []
        for i in range(self.n_flows):
            if self.conditioning_option.lower() == "parallel":
                hcond = self.conditioning_layers[i](embedding)
            elif self.conditioning_option.lower() == "sequential":
                hcond = self.conditioning_layers[i](hcond)
            hconds.append(hcond)
        # hcond : (16, 128, 1, 1)
        # hconds = [hcond * 20]
        if not reverse:
            logdet = 0.0
            for i in range(self.n_flows):
                x, logdet_ = self.sub_layers[i](x, hconds[i])  # x: (16, 128, 1, 1); hconds[i]: (16, 128, 1, 1)
                # x: (16, 128, 1, 1)  logdet_: (16,)
                logdet = logdet + logdet_
                self.last_outs.append(x)
                self.last_logdets.append(logdet)
            return x, logdet
        else:
            for i in reversed(range(self.n_flows)):
                x = self.sub_layers[i](x, hconds[i], reverse=True)
            return x

    def reverse(self, out, xcond):
        return self(out, xcond, reverse=True)


class ConditionalTransformer(nn.Module):
    """
    Conditional Invertible Neural Network.
    Can be conditioned both on input with spatial dimension (i.e. a tensor of shape BxCxHxW) and a flat input
    (i.e. a tensor of shape BxC)
    """
    def __init__(self,):
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        super().__init__()
        # get all the hyperparameters
        in_channels = 256
        mid_channels = 1024
        hidden_depth = 2
        n_flows = 24
        conditioning_option = 'None'
        flowactivation = 'none'
        embedding_channels = 256
        n_down = 5 # conv5 2

        self.emb_channels = embedding_channels  # 128
        self.in_channels = in_channels  # 128

        self.flow = ConditionalFlow(in_channels=in_channels,  # 128
                                    embedding_dim=self.emb_channels,  # 128
                                    hidden_dim=mid_channels,  # 1024
                                    hidden_depth=hidden_depth,  # 2
                                    n_flows=n_flows,  # 20
                                    conditioning_option=conditioning_option,  # None
                                    activation=flowactivation)  # None
        conditioning_spatial_size = 176 # conv5 13
        conditioning_in_channels = 1 # 1000 # conv5 256

        if conditioning_spatial_size == 1:
            depth = 4
            dims = 'none'
            dims = None if dims == "none" else dims
            self.embedder = DenseEmbedder(conditioning_in_channels,
                                          in_channels,
                                          depth=depth,
                                          given_dims=dims)
        else:
            self.embedder = Embedder(conditioning_spatial_size,  # 13
                                     conditioning_in_channels,  # 256
                                     in_channels,  # 128
                                     n_down=n_down)  # 2
        # self.embedder:
        #  Conv2d(256, 64, 4, 2, 1, b=False)
        #  ActNorm()
        #  LeakyReLU(0.2)

        #  Conv2d(64, 128, 4, 2, 1, b=False)
        #  ActNorm()
        #  LeakyReLU(0.2)

        #  Conv2d(128, 128, 3, 1, )

    def embed(self, conditioning):
        # embed it via embedding layer
        embedding = self.embedder(conditioning)
        return embedding

    def sample(self, shape, conditioning):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, self.embed(conditioning))
        return sample

    def forward(self, input, conditioning, train=False):
        # input(zae): (16, 128, 1, 1)
        # conditioning(zrep): (16, 256, 13, 13)
        embedding = self.embed(conditioning)  # (16, 128)
        out, logdet = self.flow(input, embedding)
        # out: (16, 128, 1, 1)  logdet: (16)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out, conditioning):
        embedding = self.embed(conditioning)
        return self.flow(out, embedding, reverse=True)

    def get_last_layer(self):
        return getattr(self.flow.sub_layers[-1].coupling.t[-1].main[-1], 'weight')
