"""
StudentCMABi CBM — StudentCMABi 위에 Residual CBM

전제:
  train_egemap_kd.py 실행 완료 →
    student_kd_best.pt
    student_kd_fused_expanded.npy   (N, 320)
    student_kd_logits.npy           (N, 2)

CBM 구조 (fused_expanded=320, n_concepts=7):
  final_logits = concept_classifier(c_pred) + w × base_logits

residual_w 스위프: [0.65, 0.70, 0.75, 0.80, 0.85]
λ_c          : [4.0, 8.0]

실행:
  cd presense && python ai/train/train_egemap_cbm.py
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, accuracy_score

_root = Path(__file__).resolve().parent.parent.parent
for p in [_root, _root/'ai', _root/'ai'/'models', _root/'ai'/'data', _root/'ai'/'train']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from dataset_e2e import build_sample_list
from models_e2e import CBMWithResidual, FUSED_EXP_DIM

DEVICE    = 'mps' if torch.backends.mps.is_available() else \
            'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR  = config.CHECKPOINTS_DIR / 'binary'
FINAL_DIR = CKPT_DIR / 'final'

# ── 입력 ──────────────────────────────────────────────────────────────────────
FUSED_CACHE  = CKPT_DIR / 'student_kd_fused_expanded.npy'
LOGITS_CACHE = CKPT_DIR / 'student_kd_logits.npy'
CONCEPT_PATH = CKPT_DIR / 'cbm_concepts.npy'

CONCEPT_NAMES = ['목소리 거칠기', '음색 불안정성', '음성 두께', '음성 피치',
                 '표정 변화량',   '입술 긴장도',   '눈썹 찡그림']
KEEP_IDX   = [0, 1, 2, 3, 4, 5, 6]
N_CONCEPTS = 7
FUSED_DIM  = FUSED_EXP_DIM   # 320

RESIDUAL_WEIGHTS = [0.0, 0.3, 0.5, 0.65, 0.70, 0.75, 0.80, 0.85]   # StudentCMABi logit 혼합 비율
LAMBDAS_C    = [4.0, 8.0]
SEED         = 42

torch.manual_seed(SEED); np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  단일 residual_w × λ_c 실험
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(residual_w, lambda_c,
                   fused_t, logits_t, lb_t, cl_t, tri, vi, N_all, concepts):
    print(f"\n  --- residual_w={residual_w}  λ_c={lambda_c} ---")
    cbm = CBMWithResidual(FUSED_DIM, N_CONCEPTS, residual_w=residual_w, dropout=0.3).to(DEVICE)

    # ── Stage 1: concept predictor warmup ────────────────────────────────────
    opt1   = torch.optim.Adam(cbm.parameters(), lr=1e-3, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, 150)
    best_mse, patience1, best_s1 = 1e9, 0, None
    for ep in range(1, 151):
        cbm.train()
        perm = torch.randperm(len(tri))
        for b in range(0, len(perm), 64):
            bi  = tri[perm[b:b+64]]
            _, cp = cbm(fused_t[bi], torch.zeros(len(bi), 2, device=DEVICE))
            loss  = F.mse_loss(cp, cl_t[bi])
            opt1.zero_grad(); loss.backward(); opt1.step()
        sched1.step()
        cbm.eval()
        with torch.no_grad():
            _, cp_v = cbm(fused_t[vi], torch.zeros(len(vi), 2, device=DEVICE))
            mse_v   = F.mse_loss(cp_v, cl_t[vi]).item()
        if mse_v < best_mse:
            best_mse = mse_v; patience1 = 0; best_s1 = {k: v.clone() for k, v in cbm.state_dict().items()}
        else:
            patience1 += 1
            if patience1 >= 25: break
    cbm.load_state_dict(best_s1)
    print(f"    Stage1 best_mse={best_mse:.5f}")

    # ── Stage 2: joint fine-tune ──────────────────────────────────────────────
    opt2   = torch.optim.Adam(cbm.parameters(), lr=2e-4, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, 100)

    def eval_acc():
        cbm.eval()
        with torch.no_grad():
            logits, _ = cbm(fused_t[vi], logits_t[vi])
        return accuracy_score(lb_t[vi].cpu().numpy(), logits.argmax(1).cpu().numpy())

    best_acc, best_s2, patience2 = 0.0, None, 0
    for ep in range(1, 101):
        cbm.train()
        perm = torch.randperm(len(tri))
        for b in range(0, len(perm), 32):
            bi = tri[perm[b:b+32]]
            logits, cp = cbm(fused_t[bi], logits_t[bi])
            loss = F.cross_entropy(logits, lb_t[bi]) + lambda_c * F.mse_loss(cp, cl_t[bi])
            opt2.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(cbm.parameters(), 1.0); opt2.step()
        sched2.step()
        val_acc = eval_acc()
        if val_acc > best_acc:
            best_acc = val_acc; patience2 = 0; best_s2 = {k: v.clone() for k, v in cbm.state_dict().items()}
        else:
            patience2 += 1
            if patience2 >= 25: print(f"    ES EP{ep}"); break

    cbm.load_state_dict(best_s2)

    # ── R² 평가 ──────────────────────────────────────────────────────────────
    cbm.eval()
    all_cp = []
    with torch.no_grad():
        for b in range(0, N_all, 64):
            e  = min(b + 64, N_all)
            bi = torch.arange(b, e, device=DEVICE)
            _, cp = cbm(fused_t[bi], logits_t[bi])
            all_cp.append(cp.cpu())
    c_pred_np = torch.cat(all_cp).numpy()
    r2 = r2_score(concepts, c_pred_np, multioutput='uniform_average')

    print(f"    Acc={best_acc:.4f}  R²={r2:.4f}")
    for i, cn in enumerate(CONCEPT_NAMES):
        r2i  = r2_score(concepts[:, i], c_pred_np[:, i])
        sign = '✅' if r2i >= 0 else '❌'
        print(f"      {sign} {cn:<14}  R²={r2i:+.3f}")

    tag = f"v2_w{residual_w}_lc{lambda_c}"
    torch.save({'cbm_state_dict': best_s2, 'val_acc': best_acc, 'r2': r2,
                'residual_w': residual_w, 'lambda_c': lambda_c},
               CKPT_DIR / f'cbm_bi_residual_{tag}_best.pt')
    return best_acc, r2


# ══════════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  StudentCMABi CBM")
    print(f"  residual_w={RESIDUAL_WEIGHTS}  λ_c={LAMBDAS_C}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    for p, name in [(FUSED_CACHE, 'fused_expanded'), (LOGITS_CACHE, 'logits'),
                    (CONCEPT_PATH, 'concepts')]:
        if not p.exists():
            print(f"⛔  {name} 없음: {p}")
            print("    train_egemap_kd.py 먼저 실행 필요")
            return

    fused_np   = np.load(FUSED_CACHE).astype(np.float32)
    logits_np  = np.load(LOGITS_CACHE).astype(np.float32)
    concepts   = np.load(CONCEPT_PATH)[:, KEEP_IDX].astype(np.float32)

    labels_df   = pd.read_csv(config.LABELS_FILE)
    labels_map  = dict(zip(labels_df.iloc[:, 0], labels_df.iloc[:, 1]))
    samples_all = build_sample_list(labels_map)
    all_labels  = np.array([s['label']   for s in samples_all], dtype=np.int64)
    subjects    = np.array([s['subject'] for s in samples_all])

    gss = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(gss.split(range(len(samples_all)), all_labels, subjects))

    fused_t   = torch.tensor(fused_np).to(DEVICE)
    logits_t  = torch.tensor(logits_np).to(DEVICE)
    lb_t      = torch.tensor(all_labels, dtype=torch.long).to(DEVICE)
    cl_t      = torch.tensor(concepts, dtype=torch.float32).to(DEVICE)
    tri       = torch.tensor(tr_idx)
    vi        = torch.tensor(val_idx)

    # StudentCMABi 단독 acc (캐시된 logit으로 계산)
    base_acc = accuracy_score(all_labels[val_idx], logits_np[val_idx].argmax(1))
    print(f"\n  StudentCMABi 단독: {base_acc:.4f}")

    results = []
    for lc in LAMBDAS_C:
        for w in RESIDUAL_WEIGHTS:
            acc, r2 = run_experiment(
                w, lc, fused_t, logits_t, lb_t, cl_t, tri, vi, len(all_labels), concepts)
            results.append((w, lc, acc, r2))

    print("\n" + "=" * 60)
    print("  결과 요약")
    print(f"  {'residual_w':<8} {'λ_c':<6} {'Acc':>7}  {'R²':>7}")
    print(f"  StudentCMABi 단독  {base_acc:>7.4f}  {'—':>7}")
    print(f"  이전 기준선    {'0.8099':>7}  {'0.7916':>7}")
    for w, lc, acc, r2 in sorted(results, key=lambda x: -x[2]):
        print(f"  w={w:<6} lc={lc:<4}  {acc:>7.4f}  {r2:>7.4f}")
    print("=== DONE ===")


if __name__ == '__main__':
    main()
