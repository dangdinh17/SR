from torch import nn
from torch.nn.utils import weight_norm


class ResBlockA(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0):
        super(ResBlockA, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats, n_feats * expansion_ratio, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        return x + self.module(x) * self.res_scale


class WDSR_A(nn.Module):
    def __init__(self):
        super(WDSR_A, self).__init__()
        n_feats = 32
        scale = 4
        n_res_blocks = 16
        expansion_ratio =6
        res_scale = 1
        head = [weight_norm(nn.Conv2d(3, n_feats, kernel_size=3, padding=1))]
        body = [ResBlockA(n_feats, expansion_ratio, res_scale) for _ in range(n_res_blocks)]
        tail = [weight_norm(nn.Conv2d(n_feats, 3 * (scale ** 2), kernel_size=3, padding=1)),
                nn.PixelShuffle(scale)]
        skip = [weight_norm(nn.Conv2d(3, 3 * (scale ** 2), kernel_size=5, padding=2)), nn.PixelShuffle(scale)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)


    def forward(self, x):

        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s

        return x



class ResBlockB(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0, low_rank_ratio=0.8):
        super(ResBlockB, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats, n_feats * expansion_ratio, kernel_size=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, int(n_feats * low_rank_ratio), kernel_size=1)),
            weight_norm(nn.Conv2d(int(n_feats * low_rank_ratio), n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x):
        return x + self.module(x) * self.res_scale


class WDSR_B(nn.Module):
    def __init__(self):
        super(WDSR_B, self).__init__()
        n_feats = 32
        scale = 4
        n_res_blocks = 16
        expansion_ratio =6
        res_scale = 1
        low_rank_ratio = 0.8
        head = [weight_norm(nn.Conv2d(3, n_feats, kernel_size=3, padding=1))]
        body = [ResBlockB(n_feats, expansion_ratio, res_scale, low_rank_ratio)
                for _ in range(n_res_blocks)]
        tail = [weight_norm(nn.Conv2d(n_feats, 3 * (scale ** 2), kernel_size=3, padding=1)),
                nn.PixelShuffle(scale)]
        skip = [weight_norm(nn.Conv2d(3, 3 * (scale ** 2), kernel_size=5, padding=2)), nn.PixelShuffle(scale)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)



    def forward(self, x):

        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s


        return x