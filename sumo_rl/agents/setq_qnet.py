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
    def __init__(self, header_dim, slot_dim, embed_dim=32, k_max=11, hidden=128, n_layers=1, unserved_max=False, served_max=False):
        super().__init__()
        self.header_dim, self.slot_dim, self.embed_dim, self.k_max = header_dim, slot_dim, embed_dim, k_max
        # unserved_max (2026-10-14, 用户批准 "被压运动除了取平均, 再加一个取最大"; 默认 False → 旧配置逐位不变):
        #   相位头另读被压运动编码的逐维最大值 u_max, 让单个运动上的急情 (如一辆救护车) 不被约 10 个被压运动的平均稀释。
        self.unserved_max = bool(unserved_max)
        # served_max (2026-10-14, 用户令 "你来决定" → 两侧都用 平均 + 最大; 默认 False → 旧配置逐位不变): 相位头另读放行运动编码的逐维最大值 s_max。
        #   理由: 都柏林枚举相位放行侧多为 3 个运动、被压侧 1-6 个, 两侧都会被平均稀释; 最大值不随集合大小变化, 适合相位数不同的路口共用一张网。
        self.served_max = bool(served_max)
        self.enc = torch.nn.Sequential(torch.nn.Linear(slot_dim, embed_dim), torch.nn.ReLU(), torch.nn.LayerNorm(embed_dim))
        layers, d = [], 2 * embed_dim + 2 + 1 + (embed_dim if self.unserved_max else 0) + (embed_dim if self.served_max else 0)
        for _ in range(max(int(n_layers), 1)):
            layers += [torch.nn.Linear(d, hidden), torch.nn.ReLU()]; d = hidden
        layers.append(torch.nn.Linear(d, 1))
        self.head = torch.nn.Sequential(*layers)

    def encode(self, x):
        B = x.shape[0]
        return self.enc(x[:, self.header_dim:].reshape(B, 12, self.slot_dim))                 # (B,12,E)

    def features(self, x, pm, exist):
        """(B,obs),(B,K,12),(B,12) -> (B,K,2E+3): [served mean, unserved mean, n_served/12, n_unserved/12, is_current];
        unserved_max 时再接 unserved max, served_max 时再接 served max (两者都开共 4E+3)。"""
        B = x.shape[0]
        e = self.encode(x)                                                                      # (B,12,E)
        ex = exist.unsqueeze(1)                                                                 # (B,1,12)
        sv = pm * ex; us = (1.0 - pm) * ex                                                      # (B,K,12)
        ns = sv.sum(-1, keepdim=True); nu = us.sum(-1, keepdim=True)
        s_mean = torch.bmm(sv, e) / ns.clamp(min=1.0); u_mean = torch.bmm(us, e) / nu.clamp(min=1.0)
        green = x[:, self.header_dim:].reshape(B, 12, self.slot_dim)[:, :, 0]                   # (B,12) is_green
        cur = ((pm == green.unsqueeze(1)) | (exist.unsqueeze(1) == 0)).all(-1).float().unsqueeze(-1)
        feats = [s_mean, u_mean, ns / 12.0, nu / 12.0, cur]
        if self.unserved_max:   # 被压运动编码逐维最大值; 被压集合为空的候选取 0 (用有限大负数做掩码, 不产生 NaN)
            m = (us > 0.5).unsqueeze(-1)                                                       # (B,K,12,1)
            u_max = e.unsqueeze(1).expand(-1, us.shape[1], -1, -1).masked_fill(~m, -1e4).amax(dim=2)   # (B,K,E)
            feats.append(torch.where(nu > 0, u_max, torch.zeros_like(u_max)))
        if self.served_max:     # 放行运动编码逐维最大值; 放行集合为空的候选取 0
            m = (sv > 0.5).unsqueeze(-1)
            s_max = e.unsqueeze(1).expand(-1, sv.shape[1], -1, -1).masked_fill(~m, -1e4).amax(dim=2)
            feats.append(torch.where(ns > 0, s_max, torch.zeros_like(s_max)))
        return torch.cat(feats, dim=-1)

    def forward(self, x, pm, rel, exist):
        return self.head(self.features(x, pm, exist)).squeeze(-1)                               # (B,K)
