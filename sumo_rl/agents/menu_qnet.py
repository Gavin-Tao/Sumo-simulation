"""菜单条件化 Q 网络 (2026-10-10, 用户批准; arch: menuq, 默认不启用 → 旧配置逐位不变)。

动机 (DUBLIN18H_GS_ABLATION §6): FRAP 把相位分拆成运动分相加, 且不读观测头部, 在 2 相位路口也学不出 "保持主相位";
8STD 的 MLP 看整条状态却只能用固定位置的 8 个标准相位。本网络两者兼得:
    Q(s, k) = MLP([ s (整条观测: 头部 + 12 槽) , d_k (候选相位 k 的说明书: 12 维 "放行哪些运动" 多热位) , is_current_k ])
对路口菜单里的每个候选 k 各算一次, 掩码无效行 (与 FRAPQNet 相同契约: 无效行是垃圾, 调用方必须用 mask)。
说明书用运动语言, 任意菜单、任意路口共享一张网; 整条状态让它能学 "保持" 与 "集中绿灯"; 容量与 8STD 的 MLP 同量级。
is_current_k = 1 当且仅当候选 k 的放行集合与当前 is_green 位 (perphase 槽特征 0) 在所有存在的槽上一致 (与 hold_bias 同法)。
前向签名与 FRAPQNet 相同: forward(x, pm, rel, exist) -> (B, K_max); rel 不使用 (保留签名)。
"""
import torch


class MenuQNet(torch.nn.Module):
    def __init__(self, header_dim, slot_dim, k_max=11, hidden=128, n_layers=2):
        super().__init__()
        self.header_dim, self.slot_dim, self.k_max = header_dim, slot_dim, k_max
        obs_dim = header_dim + 12 * slot_dim
        in_dim = obs_dim + 12 + 1                      # 状态 + 说明书 (12) + is_current (1)
        layers, d = [], in_dim
        for _ in range(max(int(n_layers), 1)):
            layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU()]; d = hidden
        layers.append(torch.nn.Linear(d, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def descriptors(self, x, pm, exist):
        """(B,obs),(B,K,12),(B,12) -> (B,K,13): [pm_k, is_current_k]。"""
        B = x.shape[0]
        green = x[:, self.header_dim:].reshape(B, 12, self.slot_dim)[:, :, 0]          # (B,12) is_green
        match = ((pm == green.unsqueeze(1)) | (exist.unsqueeze(1) == 0)).all(-1)        # (B,K) bool
        return torch.cat([pm, match.float().unsqueeze(-1)], dim=-1)

    def forward(self, x, pm, rel, exist):
        B, K, _ = pm.shape
        d = self.descriptors(x, pm, exist)                                              # (B,K,13)
        xs = x.unsqueeze(1).expand(B, K, x.shape[1])                                    # (B,K,obs)
        return self.mlp(torch.cat([xs, d], dim=-1)).squeeze(-1)                         # (B,K)
