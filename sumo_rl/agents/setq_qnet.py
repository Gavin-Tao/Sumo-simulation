"""集合式相位头 (arch: setq; 2026-10-14, 用户批准; 默认不启用 → 旧配置逐位不变)。

动机 (analysis/GRID1X1_F1_STATE_GRID_2026-10-13.txt §17): FRAP 的 "运动分相加 + 对决分" 在能切换时只有 14-27% 倾向保持;
保持余量分解显示对决分把当前相位压下去 ("放等得最久的" 近视规则), 加性结构表达不出依赖整体局面的保持价值。
本网络保留 FRAP 的运动编码器 (所有运动共享参数), 把相位打分换成相位级的联合函数:
    e_i = Enc(x_i)                                   每个运动槽 (与 FRAP 编码器同一形式: Linear → ReLU → LayerNorm)
    对候选相位 k:  s_k = mean_{i∈k} e_i,  u_k = mean_{i∉k} e_i   (只对存在的运动取平均)
                   n_k = (|k| / 12, |非 k| / 12),  c_k = 1 当且仅当 k 的放行集合与当前 is_green 在所有存在的槽上一致
    Q(s,k) = MLP([s_k, u_k, n_k, c_k])
不做运动分相加、没有对决项; 任意路口、任意相位集合共用一张网。需要 perphase 观测 (槽第 0 维为 is_green); 头部 (若有) 不进网络, 与 FRAP 相同。
前向签名与 FRAPQNet 相同: forward(x, pm, rel, exist) -> (B, K_max) 原始 Q (调用方负责掩码); rel 不使用。
"""
import torch


class SetQNet(torch.nn.Module):
    def __init__(self, header_dim, slot_dim, embed_dim=32, k_max=11, hidden=128, n_layers=1):
        super().__init__()
        self.header_dim, self.slot_dim, self.embed_dim, self.k_max = header_dim, slot_dim, embed_dim, k_max
        self.enc = torch.nn.Sequential(torch.nn.Linear(slot_dim, embed_dim), torch.nn.ReLU(), torch.nn.LayerNorm(embed_dim))
        layers, d = [], 2 * embed_dim + 2 + 1
        for _ in range(max(int(n_layers), 1)):
            layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU()]; d = hidden
        layers.append(torch.nn.Linear(d, 1))
        self.head = torch.nn.Sequential(*layers)

    def encode(self, x):
        B = x.shape[0]
        return self.enc(x[:, self.header_dim:].reshape(B, 12, self.slot_dim))                 # (B,12,E)

    def features(self, x, pm, exist):
        """(B,obs),(B,K,12),(B,12) -> (B,K,2E+3): [served mean, unserved mean, n_served/12, n_unserved/12, is_current]。"""
        B = x.shape[0]
        e = self.encode(x)                                                                      # (B,12,E)
        ex = exist.unsqueeze(1)                                                                 # (B,1,12)
        sv = pm * ex; us = (1.0 - pm) * ex                                                      # (B,K,12)
        ns = sv.sum(-1, keepdim=True); nu = us.sum(-1, keepdim=True)
        s_mean = torch.bmm(sv, e) / ns.clamp(min=1.0); u_mean = torch.bmm(us, e) / nu.clamp(min=1.0)
        green = x[:, self.header_dim:].reshape(B, 12, self.slot_dim)[:, :, 0]                   # (B,12) is_green
        cur = ((pm == green.unsqueeze(1)) | (exist.unsqueeze(1) == 0)).all(-1).float().unsqueeze(-1)
        return torch.cat([s_mean, u_mean, ns / 12.0, nu / 12.0, cur], dim=-1)

    def forward(self, x, pm, rel, exist):
        return self.head(self.features(x, pm, exist)).squeeze(-1)                               # (B,K)
