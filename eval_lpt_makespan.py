import sys, os, json, numpy as np, pandas as pd, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lpt'))
import torch
from model import SchedulerAgent
from train_LPT_ddp import PointerHead, _makespan_LPT, _makespan_lower_bound

DEVICE = torch.device('cuda:1')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, 'checkpoints', 'policy_lpt_best.pt')
MACHINE_COUNTS = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [5, 10, 20, 50, 100]
CHUNK = 20000    # chunk encoder forward (shared GPU with training)

def log(msg):
    print(msg, flush=True)
    with open('/root/non-clairvoyant-with-behavior/eval_lpt_makespan.log', 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} | {msg}\n')

def rl_makespan(ptr_head, EMB, _phat, true_all, n_all, m):
    """RL rollout: pick job by ptr_head, assign to least-loaded (perceived), makespan on TRUE loads."""
    n_sample = min(n_all, m * max(10, int(np.log2(m) * 10)))
    rng = np.random.RandomState(42)
    pick = np.sort(rng.choice(n_all, size=n_sample, replace=False))
    true_n = true_all[pick].astype(np.float64)
    pred_n = np.maximum(_phat[pick], 1.0)
    emb_n = EMB[pick]
    n = len(true_n)

    perceived = np.zeros(m, dtype=np.float64)
    actual = np.zeros(m, dtype=np.float64)
    assigned = np.zeros(n, dtype=bool)
    for _ in range(n):
        pending = np.where(~assigned)[0]
        if len(pending) == 0: break
        Np = len(pending)
        rt = torch.zeros(Np, 3, dtype=torch.float32, device=DEVICE)
        jb = torch.cat([emb_n[pending], rt], dim=-1).unsqueeze(0)
        gl = torch.tensor([[0.0, float(Np), 0.0]], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            scores = ptr_head(jb, gl).squeeze(0)
        jid_in_pending = scores.argmax().item()
        jid = int(pending[jid_in_pending])
        mi = int(np.argmin(perceived))
        perceived[mi] += pred_n[jid]
        actual[mi] += true_n[jid]
        assigned[jid] = True

    opt_pre = _makespan_lower_bound(true_n, m)
    return dict(rl_rho=float(np.max(actual)) / opt_pre,
                lpt_rho=_makespan_LPT(true_n, m) / opt_pre,
                n=n_sample)

def main():
    agent = SchedulerAgent(os.path.join(_ROOT, 'checkpoints', 'encoder_best.pt'), device=DEVICE)
    ptr_head = PointerHead().to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    ptr_head.load_state_dict(ckpt['ptr_head'], strict=False)
    agent.encoder.load_state_dict(ckpt['encoder'])
    ptr_head.eval(); agent.encoder.eval()

    fc = json.load(open(os.path.join(_ROOT, 'data', 'feature_config.json')))
    NUM = [f for f in fc['feats'] if f not in fc['cat_cards'] and f != 'group_enc']
    CAT = [k for k in fc['cat_cards'] if k != 'group_enc']
    df = pd.read_csv(os.path.join(_ROOT, 'data', 'test_online.csv'))
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = df[c].astype('float32')
    _xc = torch.tensor(df[NUM].values.astype(np.float32), device=DEVICE)
    _xcat = torch.tensor(df[CAT].values.astype(np.int64), device=DEVICE).clamp(min=0)
    _TRUE_ALL = df["target_p_star"].values.astype(np.float32)
    n_all = len(df)

    ck = torch.load(os.path.join(_ROOT, 'checkpoints', 'encoder_best.pt'), map_location='cpu', weights_only=False)
    Y_MU = float(ck['y_mu']); Y_SIGMA = float(ck['y_sigma'])

    EMBs, PSs = [], []
    with torch.no_grad():
        for s in range(0, n_all, CHUNK):
            e = _xc[s:s+CHUNK]; c = _xcat[s:s+CHUNK]
            EMBs.append(agent.encoder(e, c).cpu())
            PSs.append(agent.encoder.predict(e, c).squeeze().cpu())
    EMB = torch.cat(EMBs, dim=0).to(DEVICE)
    _ps = torch.cat(PSs, dim=0)
    _phat = np.expm1(_ps.numpy() * Y_SIGMA + Y_MU).astype(np.float64)

    t0 = time.time()
    log(f"=== eval_lpt_makespan | ckpt={os.path.basename(CKPT)} | test n={n_all:,} | m={MACHINE_COUNTS} | cuda:{DEVICE.index} ===")
    results = {}
    for m in MACHINE_COUNTS:
        results[m] = rl_makespan(ptr_head, EMB, _phat, _TRUE_ALL, n_all, m)
    avg = np.mean([results[m]['rl_rho'] for m in MACHINE_COUNTS])
    lpt_avg = np.mean([results[m]['lpt_rho'] for m in MACHINE_COUNTS])

    for m in MACHINE_COUNTS:
        log(f"  m={m:>3d} (n={results[m]['n']:>5d})  RL ρ={results[m]['rl_rho']:.3f}  真值LPT ρ={results[m]['lpt_rho']:.3f}")
    log(f"  avg_RL={avg:.4f}  avg_真值LPT={lpt_avg:.4f}  | 耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
