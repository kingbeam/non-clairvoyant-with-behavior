"""
DDP training of a preemptive EDF-BC policy for max-stretch.

Oracle: EDF at S* (optimal for max-stretch via bisection).
Behavior cloning (BC): learn to pick the job with the earliest deadline at each
decision point, from decision states produced by the EDF expert.

Usage:
    torchrun --nproc_per_node=<N> train_EDF_ddp.py
"""
import sys, os, time, json, numpy as np, pandas as pd, heapq, math
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as torch_checkpoint

torch.manual_seed(42); np.random.seed(42)

from model import SchedulerAgent, PointerHead

EPS = 1e-9


# ===================================================================
#  Config
# ===================================================================

CFG = dict(
    data_path    = "data/train_online.csv",
    test_path    = "data/val_online.csv",
    feature_cfg  = "data/feature_config.json",

    n_iters      = 1000,
    n_jobs       = 7000,
    batch_size   = 128,
    lr           = 2e-3,
    max_grad     = 0.5,
    aux_loss_w   = 0.1, 
)


# ===================================================================
#  DDP helpers
# ===================================================================

def setup_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0


# ===================================================================
#  EDF oracle (max-stretch optimal)
# ===================================================================

def _edf_feasible(r, p, d):
    """EDF feasibility check (preemptive uniprocessor)."""
    n = len(r); rem = p.copy()
    heap = []; i, t = 0, 0.0; iters = 0
    while i < n or heap:
        iters += 1
        if iters > n * 100:  # safety guard
            return False
        if not heap and i < n: t = max(t, r[i])
        while i < n and r[i] <= t + EPS:
            heapq.heappush(heap, (d[i], i)); i += 1
        if not heap: continue
        dj, j = heapq.heappop(heap)
        if t > dj + EPS: return False
        next_arr = r[i] if i < n else math.inf
        dt = min(rem[j], next_arr - t, dj - t)
        if dt <= 1e-12:
            t = min(next_arr, dj)
            if rem[j] > EPS: heapq.heappush(heap, (dj, j))
            continue
        rem[j] -= dt; t += dt
        if rem[j] > EPS:
            if t >= dj - EPS: return False
            heapq.heappush(heap, (dj, j))
    return True

def _opt_max_stretch(r, p, tol=1e-3, max_iter=60):
    """Bisection search for the optimal stretch S*."""
    def feasible(S): return _edf_feasible(r, p, r + S * p)
    lo, hi = 1.0, 2.0
    for _ in range(100):  # guard against infinite loop
        if feasible(hi): break
        hi *= 2.0
    else:
        return hi  # fallback
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if feasible(mid): hi = mid
        else: lo = mid
        if hi - lo <= tol * max(1.0, hi): break
    return hi


def _generate_edf_trace(arr, true, S, n):
    """Run the EDF expert at S* and record each chosen job.
    EDF picks the job with earliest deadline d_j = r_j + S* * p_j among pending jobs."""
    d = arr + S * np.maximum(true, 1.0)  # deadlines
    rem = true.copy().astype(np.float64)
    att = np.zeros(n, dtype=np.float64)
    heap = []  # (deadline, job_id)
    i_arrived = 0
    order = sorted(range(n), key=lambda j: arr[j])
    t = float(arr.min())
    trace = []
    last_running = -1
    last_update_t = t
    arrived = np.zeros(n, dtype=bool)
    pos = np.zeros(n, dtype=np.int32)

    while i_arrived < n or heap:
        if not heap and i_arrived < n:
            t = max(t, arr[order[i_arrived]])
            last_update_t = t

        while i_arrived < n and arr[order[i_arrived]] <= t + 1e-6:
            j = order[i_arrived]; arrived[j] = True
            heapq.heappush(heap, (d[j], j))
            i_arrived += 1

        if not heap: continue

        if last_running >= 0:
            att[last_running] += (t - last_update_t)
        last_update_t = t

        # EDF choice: earliest deadline
        dj, jid = heap[0]

        if jid != last_running:
            pending_mask = (rem > 1e-6) & arrived
            pending = np.where(pending_mask)[0]
            if len(pending) > 1:
                pos[pending] = np.arange(len(pending))
                chosen_idx = pos[jid]
                trace.append((chosen_idx, pending.copy(), rem[pending].copy(),
                             att[pending].copy(), last_running, t))

        heapq.heappop(heap)
        last_running = jid

        # Run until next event
        na = arr[order[i_arrived]] if i_arrived < n else float('inf')
        run = min(rem[jid], na - t, d[jid] - t)
        if rem[jid] <= EPS: run = 0.0
        t += run; rem[jid] -= run

        if rem[jid] <= 1e-6:
            rem[jid] = 0.0; last_running = -1
        else:
            heapq.heappush(heap, (d[jid], jid))

    return trace, d


# ===================================================================
#  Eval helpers (for test)
# ===================================================================

def _edf_schedule_eval(r, p, S):
    d = r + S * p; n = len(r)
    rem = p.copy(); C = np.full(n, np.nan, dtype=np.float64)
    heap = []; i, t = 0, 0.0
    while i < n or heap:
        if not heap and i < n: t = max(t, r[i])
        while i < n and r[i] <= t + EPS:
            heapq.heappush(heap, (d[i], i)); i += 1
        if not heap: continue
        dj, j = heapq.heappop(heap)
        next_arr = r[i] if i < n else math.inf
        dt = min(rem[j], next_arr - t, dj - t)
        if dt <= 1e-12:
            t = min(next_arr, dj)
            if rem[j] > EPS: heapq.heappush(heap, (dj, j))
            continue
        rem[j] -= dt; t += dt
        if rem[j] <= EPS: C[j] = t
        else: heapq.heappush(heap, (dj, j))
    return C


# ===================================================================
#  Main
# ===================================================================

def main():
    rank, world_size, local_rank = setup_ddp()
    DEVICE = torch.device(f"cuda:{local_rank}")

    with open(CFG["feature_cfg"]) as f:
        fc = json.load(f)
    NUM = [f for f in fc["feats"] if f not in fc["cat_cards"] and f != "group_enc"]
    CAT = [k for k in fc["cat_cards"] if k != "group_enc"]

    df_all = pd.read_csv(CFG["data_path"])
    for c in df_all.select_dtypes(include=[np.float64]).columns:
        df_all[c] = df_all[c].astype(np.float32)

    agent = SchedulerAgent("checkpoints/encoder_best.pt", device=f"cuda:{local_rank}", random_init=True)
    agent.unfreeze_encoder()   # random init + full lr: encoder is trained from scratch

    # de-standardization params for the encoder's p̂ output (paper Eq. J_pred)
    _ck_meta = torch.load("checkpoints/encoder_best.pt", map_location="cpu", weights_only=False)
    Y_MU = float(_ck_meta['y_mu']); Y_SIGMA = float(_ck_meta['y_sigma'])

    encoder = agent.encoder
    if CFG["aux_loss_w"] > 0:
        encoder.forward = encoder.forward_with_pred   # returns (emb, pred_std), keeps aux loss under DDP
    encoder_ddp = DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
    ptr_head = PointerHead().to(local_rank)
    ptr_head_ddp = DDP(ptr_head, device_ids=[local_rank])

    opt = torch.optim.Adam([
        dict(params=ptr_head_ddp.parameters(), lr=CFG["lr"]),
        dict(params=encoder_ddp.parameters(), lr=CFG["lr"]),
    ])
    scheduler = None

    if is_main_process(rank):
        print(f"EDF-DDP | {CFG['n_iters']} iters | n_jobs={CFG['n_jobs']} | "
              f"GPUs={world_size} | batch_size={CFG['batch_size']}×{world_size} | "
              f"preemptive | masked pool", flush=True)

    # ---- test set ----
    df_test = pd.read_csv(CFG["test_path"])
    for c in df_test.select_dtypes(include=[np.float64]).columns:
        df_test[c] = df_test[c].astype(np.float32)
    # random-sample the test set (seed 42), consistent with the SRPT trainer
    _W_SAMP = min(5000, len(df_test))
    _test_rng = np.random.RandomState(42)
    _test_pick = np.sort(_test_rng.choice(len(df_test), size=_W_SAMP, replace=False))
    test_df = df_test.iloc[_test_pick].sort_values('submit_time').reset_index(drop=True)
    _xc = torch.tensor(test_df[NUM].values.astype(np.float32), device=DEVICE)
    _xcat = torch.tensor(test_df[CAT].values.astype(np.int64), device=DEVICE).clamp(min=0)
    _TRUE = test_df["target_p_star"].values.astype(np.float32)
    _ARR = test_df["submit_time"].values.astype(np.float64); _ARR -= _ARR.min()
    _N = len(test_df)
    _ARR_gpu = torch.tensor(_ARR, dtype=torch.float32, device=DEVICE)

    def eval_on_test():
        encoder.eval() 
        with torch.no_grad():
            EMB = encoder(_xc, _xcat)
        encoder.train() 

        p_true = np.maximum(_TRUE.copy(), 1.0)
        # Preemptive RL
        t_rl = float(_ARR.min()); rem = _TRUE.copy().astype(np.float64)
        att = np.zeros(_N, dtype=np.float64); comp = np.zeros(_N, bool)
        C_rl = np.full(_N, np.nan, dtype=np.float64)
        att_gpu = torch.tensor(att, dtype=torch.float32, device=DEVICE)
        last_running = -1; last_t = t_rl

        for _ in range(_N * 5):
            arv = _ARR <= t_rl + 1e-6; pending = np.where(arv & ~comp)[0]
            if len(pending) == 0:
                na_arr = _ARR[~arv]
                if len(na_arr) == 0: break
                t_rl = max(t_rl, na_arr.min())
                arv = _ARR <= t_rl + 1e-6; pending = np.where(arv & ~comp)[0]
            if len(pending) == 0: break
            if last_running >= 0:
                dt = t_rl - last_t; att[last_running] += dt; att_gpu[last_running] += dt
            last_t = t_rl

            Np = len(pending); e_p = EMB[pending]
            wait = (t_rl - torch.tensor(_ARR[pending], dtype=torch.float32, device=DEVICE)) / 1e6
            running_mask = torch.tensor([1.0 if int(pending[k]) == last_running else 0.0 for k in range(Np)], device=DEVICE)
            rt = torch.stack([wait, att_gpu[pending] / 1e6, running_mask], dim=-1)
            jb = torch.cat([e_p, rt], dim=-1).unsqueeze(0)
            gl = torch.tensor([[t_rl/1e6, float(Np), 0.0]], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                idx = ptr_head(jb, gl).squeeze(0).argmax().item()
            jid = int(pending[idx])
            na_arr = _ARR[~arv]; na = na_arr.min() if len(na_arr) > 0 else float('inf')
            run = min(rem[jid], na - t_rl); t_rl += run; rem[jid] -= run
            if rem[jid] <= 1e-6:
                rem[jid] = 0.0; comp[jid] = True; C_rl[jid] = t_rl
                if last_running == jid: last_running = -1
            else: last_running = jid

        # Compute OPT (EDF at S*)
        S_star = _opt_max_stretch(_ARR, p_true)
        C_opt = _edf_schedule_eval(_ARR, p_true, S_star)

        # Metrics
        def stretch(r, C, p):
            return np.maximum((C - r) / np.maximum(p, 1e-9), 1.0)
        s_opt = stretch(_ARR, C_opt, p_true)
        s_rl = stretch(_ARR, C_rl, p_true)
        opt_max = float(np.max(s_opt))
        # ρ_max / Semp, others / OPT's own percentile/median
        rho_max = float(np.max(s_rl)) / opt_max
        rho_p99 = float(np.percentile(s_rl, 99)) / float(np.percentile(s_opt, 99))
        rho_med = float(np.median(s_rl)) / float(np.median(s_opt))

        # Also report ∑Cj (BC/SRPT)
        srpt_sum = sum(C_rl)  # our RL's sum of completion times
        # SRPT clairvoyant sum
        _rem = _TRUE.copy().astype(np.float64); _t = float(_ARR.min()); _sc_srpt = 0.0
        _heap = []; _i = 0; _order = sorted(range(_N), key=lambda j: _ARR[j])
        while _i < _N or _heap:
            if not _heap and _i < _N: _t = max(_t, _ARR[_order[_i]])
            while _i < _N and _ARR[_order[_i]] <= _t + 1e-6:
                heapq.heappush(_heap, (_rem[_order[_i]], _order[_i])); _i += 1
            if not _heap: continue
            _rj, _jid = heapq.heappop(_heap)
            _na = _ARR[_order[_i]] if _i < _N else float('inf')
            _run = min(_rj, _na - _t); _t += _run; _rj -= _run
            if _rj <= 1e-6: _sc_srpt += _t
            else: heapq.heappush(_heap, (_rj, _jid))
        bc_srpt = srpt_sum / _sc_srpt

        return dict(rho_max=rho_max, rho_p99=rho_p99, rho_med=rho_med,
                    bc_srpt=bc_srpt, S_star=float(S_star))

    best_rho_max, t0, eval_cnt = float("inf"), time.time(), 0

    for it in range(CFG["n_iters"]):
        seed = 9000 + it * world_size + rank
        start = np.random.RandomState(seed).randint(0, max(1, len(df_all) - CFG["n_jobs"]))
        ep = df_all.iloc[start:start + CFG["n_jobs"]].sort_values('submit_time').reset_index(drop=True)
        n = len(ep)
        xc = torch.tensor(ep[NUM].values.astype(np.float32), device=DEVICE)
        xcat = torch.tensor(ep[CAT].values.astype(np.int64), device=DEVICE).clamp(min=0)
        true = ep["target_p_star"].values.astype(np.float32)
        arr = ep["submit_time"].values.astype(np.float64); arr -= arr.min()
        p_true = np.maximum(true, 1.0)

        # Compute S* for this time-slice (EDF bisection)
        S_train = _opt_max_stretch(arr, p_true)

        # Generate EDF trace
        trace, _ = _generate_edf_trace(arr, true, S_train, n)

        # Encoder forward (embedding + optional prediction head for aux loss)
        out = encoder_ddp(xc, xcat)
        if isinstance(out, tuple):
            emb_full, pred_std = out
            # auxiliary prediction-head MSE in standardized log1p space
            y_std = (torch.log1p(torch.tensor(true, dtype=torch.float32, device=DEVICE)).unsqueeze(-1) - Y_MU) / Y_SIGMA
            aux_loss = F.mse_loss(pred_std, y_std) * CFG["aux_loss_w"]
        else:
            emb_full = out
            aux_loss = torch.zeros((), device=DEVICE)

        # Collect BC states
        states, targets = [], []
        for chosen_idx, pending, pending_rem, pending_att, running_jid, t in trace:
            Np = len(pending)
            e_emb = emb_full[pending]
            arr_p = torch.tensor(arr[pending], dtype=torch.float32, device=DEVICE)
            att_p = torch.tensor(pending_att, dtype=torch.float32, device=DEVICE) / 1e6
            wait = (t - arr_p) / 1e6
            running_p = torch.zeros(Np, device=DEVICE)
            if running_jid >= 0:
                rpos = np.where(pending == running_jid)[0]
                if len(rpos) > 0: running_p[rpos[0]] = 1.0
            rt = torch.stack([wait, att_p, running_p], dim=-1)
            gl = torch.tensor([[t/1e6, float(Np), 0.0]], dtype=torch.float32, device=DEVICE)
            states.append((e_emb, rt, gl, Np))
            targets.append(chosen_idx)

        # Eval every iteration
        if is_main_process(rank):
            eval_cnt += 1
            m = eval_on_test()
            if np.isfinite(m["rho_max"]) and m["rho_max"] < best_rho_max:
                best_rho_max = m["rho_max"]
                torch.save(dict(ptr_head=ptr_head_ddp.module.state_dict(),
                                encoder=encoder_ddp.module.state_dict()),
                           "checkpoints/policy_edf_best.pt")
            flag = "BEST" if np.isfinite(m["rho_max"]) and m["rho_max"] <= best_rho_max else " "
            line = (f"[#{eval_cnt}] Iter {it:>4d} | ρ_max={m['rho_max']:.3f}  "
                    f"ρ_p99={m['rho_p99']:.3f}  ρ_med={m['rho_med']:.3f}  "
                    f"BC/SRPT={m['bc_srpt']:.3f}  S*={m['S_star']:.1f}  {flag} | Time={time.time()-t0:.0f}s")
            print(line, flush=True)
            with open("eval_edf.log", "a") as f:
                f.write(line + "\n")

        if len(states) < 4: continue

        # BC training
        idx = np.random.permutation(len(states))
        losses = []
        for start_b in range(0, len(states), CFG["batch_size"]):
            bidx = idx[start_b:start_b + CFG["batch_size"]]
            max_n = max(states[i][3] for i in bidx)
            jb_b, gl_b, mask_b, tgt_b = [], [], [], []
            for i in bidx:
                e_emb, rt, gl, Np = states[i]
                jb = torch.cat([e_emb, rt], dim=-1).unsqueeze(0)
                jb_b.append(torch.cat([jb.squeeze(0), torch.zeros(max_n - Np, jb.shape[2], device=DEVICE)]))
                gl_b.append(gl)
                m = torch.zeros(max_n, device=DEVICE); m[:Np] = 1.0; mask_b.append(m)
                tgt_b.append(targets[i])
            jb_b = torch.stack(jb_b); gl_b = torch.stack(gl_b).squeeze(1)
            masks = torch.stack(mask_b)
            tgt_b = torch.tensor(tgt_b, dtype=torch.long, device=DEVICE)
            logits = torch_checkpoint(ptr_head_ddp, jb_b, gl_b, masks, use_reentrant=False)
            logits[masks == 0] = -1e9
            ce = F.cross_entropy(logits, tgt_b)   # clone the expert's chosen job
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            alpha = 0.5
            losses.append(ce - alpha * entropy)

        opt.zero_grad()
        total = sum(losses) + aux_loss   # clone loss + weighted prediction loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(ptr_head_ddp.parameters(), CFG["max_grad"])
        torch.nn.utils.clip_grad_norm_(encoder_ddp.parameters(), CFG["max_grad"])
        opt.step()
        total_loss_val = sum(l.item() for l in losses)
        avg_loss = total_loss_val / len(losses)
        if is_main_process(rank) and it % 100 == 0:
            print(f"  [rank{rank}] loss={avg_loss:.4f} (avg/batch)  states={len(states)}  lr={CFG['lr']:.2e}", flush=True)

    if is_main_process(rank):
        m = eval_on_test()
        if np.isfinite(m["rho_max"]) and m["rho_max"] < best_rho_max:
            best_rho_max = m["rho_max"]
        print(f"\nFinal: ρ_max={m['rho_max']:.3f}  ρ_p99={m['rho_p99']:.3f}  ρ_med={m['rho_med']:.3f}  BC/SRPT={m['bc_srpt']:.3f}  best={best_rho_max:.3f}", flush=True)
        torch.save(dict(ptr_head=ptr_head_ddp.module.state_dict(),
                        encoder=encoder_ddp.module.state_dict()),
                   "checkpoints/policy_edf_final.pt")
        print("Model saved to checkpoints/policy_edf_final.pt", flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
