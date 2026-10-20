"""BC 分类头的推理解码头(寄居 fork 命名空间的理由:

PPO CnnPolicy 的 action_net 只能是 Linear(512,2),而分类头(21+6 logits)无法
折叠进线性层。方案:把「softmax 期望解码」封装为 nn.Module,赋值给
policy.action_net——predict 路径 mean=action_net(latent) 原样可用。
类放在 gym_donkeycar 命名空间,SB3 save/load 的 pickle 按限定名解析,
任何 import 了 gym_donkeycar 的进程(如 rl_eval)零改动可加载。
"""
import torch
import torch.nn as nn


class ExpectDecode(nn.Module):
    """双头分箱分类 + softmax 期望解码 → 连续 (steer, throttle)。

    steer 21 箱中心 linspace(-1,1,21);throttle 6 箱中心 linspace(0,1,6)。
    """

    def __init__(self, in_features=512, n_steer=21, n_thr=6):
        super().__init__()
        self.head_s = nn.Linear(in_features, n_steer)
        self.head_t = nn.Linear(in_features, n_thr)
        self.register_buffer("centers_s", torch.linspace(-1.0, 1.0, n_steer))
        self.register_buffer("centers_t", torch.linspace(0.0, 1.0, n_thr))

    def forward(self, latent):
        ps = torch.softmax(self.head_s(latent), dim=-1)
        pt = torch.softmax(self.head_t(latent), dim=-1)
        s = (ps * self.centers_s).sum(dim=-1)
        t = (pt * self.centers_t).sum(dim=-1)
        return torch.stack([s, t], dim=-1)

    def logits(self, latent):
        return self.head_s(latent), self.head_t(latent)
