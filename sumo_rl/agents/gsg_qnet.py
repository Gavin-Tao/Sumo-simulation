"""GS-G (2026-10-12, 用户批准; 设计稿 experiments/analysis/GS_G_FINAL_DESIGN_2026-10-12.txt).

原 FRAP 学习打分 (服务分 + 对决压制分) + 两个按运动累加的切换代价:
    Q(s,k) = FRAPcore(s)_k
             - sum_{i in cur\\k} kappa_cut(e_i, gt_i, tau, mg)     截断: 现在绿、换到 k 变红的运动
             - sum_{i in k\\cur} kappa_new(e_i, tau, mg)           起动: 现在红、换到 k 变绿的运动
- cur 不做 "相位行匹配", 直接用 12 个 is_green 位: cut_ik = green_i (1 - pm_ki) exist_i, new_ik = pm_ki (1 - green_i) exist_i;
  k 与当前绿灯集合相同时两项代价都为 0 (保持不付代价)。
- tau = 相位已持续/100, mg = min_green_ok: 观测头部两维 (与 8STD 读的完全相同), 只进代价项, 不进运动编码器。
- gt_i = 运动 i 的连续放行时长 (观测槽的最后一维, 需 obs_green_elapsed): 只进 kappa_cut, 从编码器输入中拆出。
- kappa = softplus(MLP), 末层偏置初始化 -4 (约 0.018) -> 训练起点约等于原 FRAP。
与 FRAPAgent 的接口相同: forward(x, pm, rel, exist) -> (B, K_max) raw Q (调用方负责掩码)。
time_inputs (2026-10-13, 用户批准, 默认 "full" → 上述行为逐位不变):
  "none" = 代价项不读任何时间量: κ_cut(e_i), κ_new(e_i); 不要求头部与逐运动放行时长, 编码器读完整的槽
           (用于与纯 FRAP 观测逐位相同的对照, 如槽内含 is_green + 槽内 min_green_ok 的 12 槽状态)。
"""
import torch

from .frap_agent import FRAPQNet


class GSGQNet(torch.nn.Module):
    def __init__(self, header_dim, slot_dim, embed_dim=16, pair_dim=16, k_max=11,
                 gt_in_slot=True, gt_clip=1.2, cost_hidden=8, cost_sign="nonneg", time_inputs="full"):
        super().__init__()
        if time_inputs not in ("full", "none"):
            raise ValueError(f"time_inputs must be full|none, got {time_inputs}")
        self.time_inputs = time_inputs
        if time_inputs == "none":
            self._init_no_time(header_dim, slot_dim, embed_dim, pair_dim, k_max, cost_hidden, cost_sign)
            return
        if header_dim != 2:
            raise ValueError("GS-G 需要头部两维 [min_green_ok, 已持续时间] (obs_phase_state perphase, obs_header true)")
        if not gt_in_slot:
            raise ValueError("GS-G 需要逐运动放行时长 (obs_green_elapsed: true), 位于每槽最后一维")
        self.header_dim, self.slot_dim, self.k_max = header_dim, slot_dim, k_max
        self.gt_clip = float(gt_clip)
        # cost_sign (2026-10-12 消融, 默认 "nonneg" 不变): "nonneg" = softplus (切换只有代价); "free" = 不加约束 (可为负), 末层零初始化 → 起点代价 0。
        if cost_sign not in ("nonneg", "free"):
            raise ValueError(f"cost_sign must be nonneg|free, got {cost_sign}")
        self.cost_sign = cost_sign
        self.core_slot = slot_dim - 1                       # 编码器看不到 gt_i
        self.core = FRAPQNet(0, self.core_slot, embed_dim, pair_dim, k_max)
        def _cost(in_dim):
            m = torch.nn.Sequential(torch.nn.Linear(in_dim, cost_hidden), torch.nn.ReLU(), torch.nn.Linear(cost_hidden, 1))
            if cost_sign == "nonneg":
                torch.nn.init.constant_(m[2].bias, -4.0)
            else:
                torch.nn.init.zeros_(m[2].weight); torch.nn.init.zeros_(m[2].bias)
            return m
        self.kappa_cut = _cost(embed_dim + 3)               # [e_i, gt_i, tau, mg]
        self.kappa_new = _cost(embed_dim + 2)               # [e_i, tau, mg]

    def _init_no_time(self, header_dim, slot_dim, embed_dim, pair_dim, k_max, cost_hidden, cost_sign):
        """time_inputs="none": 编码器读完整的槽, 两个代价项只读 e_i; 头部 (若有) 不进网络。"""
        if cost_sign not in ("nonneg", "free"):
            raise ValueError(f"cost_sign must be nonneg|free, got {cost_sign}")
        self.header_dim, self.slot_dim, self.k_max = header_dim, slot_dim, k_max
        self.cost_sign, self.gt_clip = cost_sign, None
        self.core_slot = slot_dim
        self.core = FRAPQNet(0, self.core_slot, embed_dim, pair_dim, k_max)
        def _cost(in_dim):
            m = torch.nn.Sequential(torch.nn.Linear(in_dim, cost_hidden), torch.nn.ReLU(), torch.nn.Linear(cost_hidden, 1))
            if cost_sign == "nonneg":
                torch.nn.init.constant_(m[2].bias, -4.0)
            else:
                torch.nn.init.zeros_(m[2].weight); torch.nn.init.zeros_(m[2].bias)
            return m
        self.kappa_cut = _cost(embed_dim)                   # [e_i]
        self.kappa_new = _cost(embed_dim)                   # [e_i]

    def _split(self, x):
        B = x.shape[0]
        if self.time_inputs == "none":
            return x[:, self.header_dim:].reshape(B, 12, self.slot_dim), None
        slots = x[:, self.header_dim:].reshape(B, 12, self.slot_dim)
        feat = slots[:, :, :self.core_slot]                 # (B,12,SD-1), 第 0 维 is_green
        gt = slots[:, :, self.core_slot].clamp(max=self.gt_clip)
        return feat, gt

    def costs(self, x):
        """(B,obs) -> kappa_cut (B,12), kappa_new (B,12), green (B,12)。"""
        B = x.shape[0]
        feat, gt = self._split(x)
        e = self.core.encode(feat.reshape(B, 12 * self.core_slot))          # (B,12,E)
        act = torch.nn.functional.softplus if self.cost_sign == "nonneg" else (lambda z: z)
        if self.time_inputs == "none":
            return act(self.kappa_cut(e)).squeeze(-1), act(self.kappa_new(e)).squeeze(-1), feat[:, :, 0]
        hdr = x[:, :2]                                                      # [mg, tau]
        tau = hdr[:, 1:2].unsqueeze(1).expand(B, 12, 1); mg = hdr[:, 0:1].unsqueeze(1).expand(B, 12, 1)
        act = torch.nn.functional.softplus if self.cost_sign == "nonneg" else (lambda z: z)
        k_cut = act(self.kappa_cut(torch.cat([e, gt.unsqueeze(-1), tau, mg], -1))).squeeze(-1)
        k_new = act(self.kappa_new(torch.cat([e, tau, mg], -1))).squeeze(-1)
        return k_cut, k_new, feat[:, :, 0]

    def forward(self, x, pm, rel, exist):
        B = x.shape[0]
        feat, _ = self._split(x)
        q = self.core(feat.reshape(B, 12 * self.core_slot), pm, rel, exist)  # (B,K) 服务分 + 对决压制分
        k_cut, k_new, green = self.costs(x)
        g = green.unsqueeze(1); ex = exist.unsqueeze(1)
        cut = g * (1.0 - pm) * ex                                            # (B,K,12)
        new = pm * (1.0 - g) * ex
        return q - (cut * k_cut.unsqueeze(1)).sum(-1) - (new * k_new.unsqueeze(1)).sum(-1)
