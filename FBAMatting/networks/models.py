import networks.layers_WS as L
import networks.resnet_bn as resnet_bn
import networks.resnet_GN_WS as resnet_GN_WS
import torch
import torch.nn as nn


def build_model(args):
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=args.encoder)

    if "BN" in args.encoder:
        batch_norm = True
    else:
        batch_norm = False
    net_decoder = builder.build_decoder(arch=args.decoder, batch_norm=batch_norm)

    model = MattingModule(net_encoder, net_decoder)

    if args.weights != "default":
        sd = torch.load(args.weights)
        model.load_state_dict(sd, strict=True)

    return model


class MattingModule(nn.Module):
    def __init__(self, net_enc, net_dec):
        super(MattingModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec

    def forward(self, image, two_chan_trimap, image_n, trimap_transformed):
        resnet_input = torch.cat((image_n, trimap_transformed, two_chan_trimap), 1)
        conv_out, indices = self.encoder(resnet_input, return_feature_maps=True)
        return self.decoder(conv_out, image, indices, two_chan_trimap)


class ModelBuilder:
    def build_encoder(self, arch="resnet50_GN"):
        if arch == "resnet50_GN_WS":
            orig_resnet = resnet_GN_WS.__dict__["l_resnet50"]()
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == "resnet50_BN":
            orig_resnet = resnet_bn.__dict__["l_resnet50"]()
            net_encoder = ResnetDilatedBN(orig_resnet, dilate_scale=8)

        else:
            raise Exception("Architecture undefined!")

        num_channels = 3 + 6 + 2

        if num_channels > 3:
            print(f"modifying input layer to accept {num_channels} channels")
            net_encoder_sd = net_encoder.state_dict()
            conv1_weights = net_encoder_sd["conv1.weight"]

            c_out, c_in, h, w = conv1_weights.size()
            conv1_mod = torch.zeros(c_out, num_channels, h, w)
            conv1_mod[:, :3, :, :] = conv1_weights

            conv1 = net_encoder.conv1
            conv1.in_channels = num_channels
            conv1.weight = torch.nn.Parameter(conv1_mod)

            net_encoder.conv1 = conv1

            net_encoder_sd["conv1.weight"] = conv1_mod

            net_encoder.load_state_dict(net_encoder_sd)
        return net_encoder

    def build_decoder(self, arch="fba_decoder", batch_norm=False):
        if arch == "fba_decoder":
            net_decoder = fba_decoder(batch_norm=batch_norm)

        return net_decoder


class ResnetDilatedBN(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilatedBN, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convolutions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = [x]
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out, indices
        return [x]


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convolutions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = [x]
        x = self.relu(self.bn1(self.conv1(x)))
        conv_out.append(x)
        x, indices = self.maxpool(x)
        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out, indices
        return [x]


def norm(dim, bn=False):
    if bn is False:
        return nn.GroupNorm(32, dim)
    else:
        return nn.BatchNorm2d(dim)


def fba_fusion(alpha, img, F, B):
    F = alpha * img + (1 - alpha ** 2) * F - alpha * (1 - alpha) * B
    B = (1 - alpha) * img + (2 * alpha - alpha ** 2) * B - alpha * (1 - alpha) * F

    F = torch.clamp(F, 0, 1)
    B = torch.clamp(B, 0, 1)
    la = 0.1
    alpha = (alpha * la + torch.sum((img - B) * (F - B), 1, keepdim=True)) / (
        torch.sum((F - B) * (F - B), 1, keepdim=True) + la
    )
    alpha = torch.clamp(alpha, 0, 1)
    return alpha, F, B


class fba_decoder(nn.Module):
    def __init__(self, batch_norm=False):
        super(fba_decoder, self).__init__()
        pool_scales = (1, 2, 3, 6)
        self.batch_norm = batch_norm

        self.ppm = []

        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    L.Conv2d(2048, 256, kernel_size=1, bias=True),
                    norm(256, self.batch_norm),
                    nn.LeakyReLU(),
                ),
            )
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_up1 = nn.Sequential(
            L.Conv2d(
                2048 + len(pool_scales) * 256, 256, kernel_size=3, padding=1, bias=True,
            ),
            norm(256, self.batch_norm),
            nn.LeakyReLU(),
            L.Conv2d(256, 256, kernel_size=3, padding=1),
            norm(256, self.batch_norm),
            nn.LeakyReLU(),
        )

        self.conv_up2 = nn.Sequential(
            L.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=True),
            norm(256, self.batch_norm),
            nn.LeakyReLU(),
        )
        if self.batch_norm:
            d_up3 = 128
        else:
            d_up3 = 64
        self.conv_up3 = nn.Sequential(
            L.Conv2d(256 + d_up3, 64, kernel_size=3, padding=1, bias=True),
            norm(64, self.batch_norm),
            nn.LeakyReLU(),
        )

        self.unpool = nn.MaxUnpool2d(2, stride=2)

        self.conv_up4 = nn.Sequential(
            nn.Conv2d(64 + 3 + 3 + 2, 32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(16, 7, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, conv_out, img, indices, two_chan_trimap):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(
                nn.functional.interpolate(
                    pool_scale(conv5),
                    (input_size[2], input_size[3]),
                    mode="bilinear",
                    align_corners=False,
                ),
            )
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_up1(ppm_out)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False,
        )

        x = torch.cat((x, conv_out[-4]), 1)

        x = self.conv_up2(x)
        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False,
        )

        x = torch.cat((x, conv_out[-5]), 1)
        x = self.conv_up3(x)

        x = torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False,
        )
        x = torch.cat((x, conv_out[-6][:, :3], img, two_chan_trimap), 1)

        output = self.conv_up4(x)

        alpha = torch.clamp(output[:, 0][:, None], 0, 1)
        F = torch.sigmoid(output[:, 1:4])
        B = torch.sigmoid(output[:, 4:7])

        # FBA Fusion
        alpha, F, B = fba_fusion(alpha, img, F, B)

        output = torch.cat((alpha, F, B), 1)

        return output
