# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="bn", dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        if norm == "bn":
            self.n1 = nn.BatchNorm2d(out_ch)
            self.n2 = nn.BatchNorm2d(out_ch)
        elif norm == "gn":
            self.n1 = nn.GroupNorm(8, out_ch)
            self.n2 = nn.GroupNorm(8, out_ch)
        else:
            self.n1 = nn.Identity()
            self.n2 = nn.Identity()

        self.act = nn.ReLU(inplace=True)
        
        # Dropout2d suppresses complete feature channels during training.
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.act(self.n1(self.conv1(x)))
        
        # Apply dropout between the two convolutions.
        out = self.dropout(out)
        
        out = self.n2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = self.act(out + identity)
        return out


class AttentionGate(nn.Module):
    """
    Classic Attention U-Net gate:
    - g: decoder feature (coarser)
    - x: encoder skip feature (finer)
    returns: gated skip
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # align spatial
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm="bn", use_att=True, dropout_p=0.0):
        super().__init__()
        self.use_att = use_att

        # upsample first: in_ch -> out_ch
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # The attention gate receives the decoder feature after upsampling.
        if use_att:
            self.att = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=max(out_ch // 2, 8))
        else:
            self.att = None

        # Forward the dropout probability to the residual block.
        self.res = ResBlock(out_ch + skip_ch, out_ch, norm=norm, dropout_p=dropout_p)

    def forward(self, x, skip):
        x = self.up(x)

        # align spatial
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # gate skip by decoder x
        if self.use_att:
            skip = self.att(x, skip)

        x = torch.cat([skip, x], dim=1)
        x = self.res(x)
        return x


class AttentionResUNet(nn.Module):
    """
    Attention + Res blocks U-Net for binary segmentation.
    Added Dropout support.
    """
    def __init__(self, in_channels: int = 4, base: int = 32, norm: str = "bn",
                 use_att: bool = True, deep_supervision: bool = False, aux_heads: bool = False,
                 dropout_p: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.base = base
        self.deep_supervision = bool(deep_supervision)
        self.aux_heads = bool(aux_heads)
        self.dropout_p = dropout_p

        # Use the same dropout probability throughout the encoder.
        self.enc1 = ResBlock(in_channels, base, norm=norm, dropout_p=dropout_p)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResBlock(base, base*2, norm=norm, dropout_p=dropout_p)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResBlock(base*2, base*4, norm=norm, dropout_p=dropout_p)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResBlock(base*4, base*8, norm=norm, dropout_p=dropout_p)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ResBlock(base*8, base*16, norm=norm, dropout_p=dropout_p)

        # Use the same dropout probability throughout the decoder.
        self.up4 = UpBlock(base*16, base*8, base*8, norm=norm, use_att=use_att, dropout_p=dropout_p)
        self.up3 = UpBlock(base*8, base*4, base*4, norm=norm, use_att=use_att, dropout_p=dropout_p)
        self.up2 = UpBlock(base*4, base*2, base*2, norm=norm, use_att=use_att, dropout_p=dropout_p)
        self.up1 = UpBlock(base*2, base, base, norm=norm, use_att=use_att, dropout_p=dropout_p)

        # main segmentation head (full-res)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

        # deep supervision heads
        if self.deep_supervision:
            self.ds3 = nn.Conv2d(base*8, 1, kernel_size=1)
            self.ds2 = nn.Conv2d(base*4, 1, kernel_size=1)
            self.ds1 = nn.Conv2d(base*2, 1, kernel_size=1)

        # auxiliary heads
        if self.aux_heads:
            self.edge_head = nn.Conv2d(base, 1, kernel_size=1)
            self.sdf_head  = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        x = self.pool1(s1)

        s2 = self.enc2(x)
        x = self.pool2(s2)

        s3 = self.enc3(x)
        x = self.pool3(s3)

        s4 = self.enc4(x)
        x = self.pool4(s4)

        x = self.bottleneck(x)

        d3 = self.up4(x, s4)
        d2 = self.up3(d3, s3)
        d1 = self.up2(d2, s2)
        d0 = self.up1(d1, s1)

        p0 = self.head(d0)

        if not self.deep_supervision:
            seg_out = p0
        else:
            p3 = self.ds3(d3)
            p2 = self.ds2(d2)
            p1 = self.ds1(d1)
            seg_out = [p3, p2, p1, p0]

        if not self.aux_heads:
            return seg_out

        edge_logits = self.edge_head(d0)
        sdf_pred = self.sdf_head(d0)

        return {"seg": seg_out, "edge": edge_logits, "sdf": sdf_pred}
