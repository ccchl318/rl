"""探针：策略对参数扰动的敏感度（argmax 翻转率 vs 相对扰动幅度）。"""
import numpy as np, torch
from ppo import ActorCritic
from game2048_env_fast import FastGame2048Batch

CKPT = "results/budget_extension/epochs10_lr625e4_s2/best.pt"
model = ActorCritic(hidden=256).to("cuda")
sd = torch.load(CKPT, map_location="cuda", weights_only=False)
model.load_state_dict(sd["model"])
model.eval()

# 采样真实对局中的 4096 个棋盘（含大量拥挤态）
env = FastGame2048Batch(1024, seed=4242)
boards = []
for t in range(4):
    obs = torch.from_numpy(env.obs).to("cuda")
    mask = torch.from_numpy(env.mask).to("cuda")
    act, _, _ = model.act_and_value(obs, mask, deterministic=True)
    boards.append(env.obs.copy()); boards.append(env.mask.copy())
    env.step(act.cpu().numpy())
obs_s = torch.from_numpy(np.concatenate(boards[0::2])[:4096]).to("cuda")
mask_s = torch.from_numpy(np.concatenate(boards[1::2])[:4096]).to("cuda")

@torch.no_grad()
def actions(m):
    lg = m.pi_net(obs_s)
    lg = torch.where(mask_s, lg, torch.full_like(lg, -1e8))
    return lg.argmax(-1)

base_act = actions(model)
margins = []
with torch.no_grad():
    lg = model.pi_net(obs_s)
    lg = torch.where(mask_s, lg, torch.full_like(lg, -1e8))
    top2 = lg.topk(2, dim=-1).values
    margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()

print(f"棋盘数 {obs_s.shape[0]} | 最优-次优 logit 差距：中位数 {np.median(margins):.3f}, "
      f"均值 {margins.mean():.3f}, <0.1 的比例 {(margins < 0.1).mean():.1%}")
print()
print(f"{'相对扰动':>10} {'argmax 翻转率':>14}")
print('-' * 28)
orig = {k: v.clone() for k, v in model.state_dict().items()}
rng = np.random.default_rng(0)
for rel in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
    flips = []
    for _ in range(3):
        perturbed = {}
        for k, v in orig.items():
            if v.dtype.is_floating_point:
                scale = v.float().std().item() + 1e-12
                perturbed[k] = v + torch.randn_like(v.float(), generator=None) * scale * rel
            else:
                perturbed[k] = v
        model.load_state_dict(perturbed)
        flips.append((actions(model) != base_act).float().mean().item())
    print(f'{rel:>10.0e} {np.mean(flips):>13.1%}')
model.load_state_dict(orig)
