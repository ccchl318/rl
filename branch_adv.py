"""状态级 GRPO 分支采样：同一拥挤状态采 G 条策略续局，组内相对优势。

动机：on-policy 每状态只有一条按当前策略采样的轨迹，无法提供"同一局面
不同动作"的反事实对比；学习所需的期望回报与单轨迹数据之间存在结构性
错配。本模块在每个拥挤状态强制分支 G 条续路（首动作按策略采样、之后
按策略续 H 步），用组内均值做基线、组内标准差归一，直接制造对比信号。

训练行只取每个分支的**第一动作**转移（组优势赋给首步），后续步骤仅用于
估计该分支的回报 R = sum(gamma^h * r_h) + gamma^H * V(s_H)。已死亡的分支
其 R 即死亡折扣回报（合法的低价值样本）。行数恰为 K*G，便于与主 GAE
批次拼接后保持 CUDA Graph 的静态形状。
"""
import numpy as np
import torch

from game2048_env_fast import FastGame2048Batch


@torch.no_grad()
def collect_branch_batch(model, cells_list, branch_g, branch_h, device, gamma):
    """cells_list: list of (9,) int64 等级数组。返回 cpu 张量 dict
    （obs/mask/act/logp/adv/ret），行数 = K*G。"""
    K = len(cells_list)
    N = K * branch_g
    env = FastGame2048Batch(N)
    idx = np.arange(N)
    cells_arr = np.stack(cells_list)
    env.set_states(idx, cells_arr)

    obs = torch.from_numpy(env.obs).to(device)
    mask = torch.from_numpy(env.mask).to(device)
    act, logp, v = model.act_and_value(obs, mask)
    act0, logp0, v0 = act.cpu(), logp.cpu(), v.cpu()

    R = np.zeros(N, dtype=np.float64)
    disc = np.ones(N, dtype=np.float64)
    alive = np.ones(N, dtype=bool)
    for _ in range(branch_h):
        rew, done = env.step(act.cpu().numpy())
        R += disc * alive * rew
        disc *= gamma
        alive &= (done == 0)
        obs = torch.from_numpy(env.obs).to(device)
        mask = torch.from_numpy(env.mask).to(device)
        act, logp, v = model.act_and_value(obs, mask)
        act_l_last, logp_l_last, v_l_last = act.cpu(), logp.cpu(), v.cpu()

    # bootstrap：仅对存活分支加 V(s_H)（死亡分支的 R 即死亡折扣回报）
    R += disc * alive * v_l_last.numpy()

    Rm = R.reshape(K, branch_g)
    adv = ((Rm - Rm.mean(axis=1, keepdims=True))
           / (Rm.std(axis=1, keepdims=True) + 1e-6)).reshape(-1)
    return {
        "obs": obs0,
        "mask": mask0,
        "act": act0,
        "logp": logp0,
        "adv": torch.from_numpy(adv.astype(np.float32)),
        "ret": (adv.float() + v0),
    }
