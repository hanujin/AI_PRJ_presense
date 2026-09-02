"""
StudentCMABi KD 학습 파이프라인

  기존 Teacher 체크포인트를 로드 후 StudentCMABi를 Knowledge Distillation으로 학습:
  - eGeMAPS late fusion (fused_expanded 이후)
  - KD Loss: L = γ×CE + α×KL(s/T ‖ t/T)×T² + β×MSE(fused_s, fused_t)
  - physio 없는 샘플은 KL·MSE 항 제외 (pmask)

결과물:
  checkpoints/binary/final/student_kd_best.pt
  checkpoints/binary/student_kd_fused_expanded.npy   (N, 320)

실행:
  cd presense && python ai/train/train_egemap_kd.py
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

_root = Path(__file__).resolve().parent.parent.parent
for p in [_root, _root/'ai', _root/'ai'/'models', _root/'ai'/'data', _root/'ai'/'train']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
from dataset_e2e import build_sample_list, TASK_TO_ID
from models_e2e import (
    StudentCMABi, Teacher,
    GEO_INPUT_DIM, FUSED_EXP_DIM, EGEMAP_DIM,
)

DEVICE    = 'mps' if torch.backends.mps.is_available() else \
            'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR  = config.CHECKPOINTS_DIR / 'binary'
FINAL_DIR = CKPT_DIR / 'final'
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ── 입력 캐시 ──────────────────────────────────────────────────────────────────
RESNET_CACHE = CKPT_DIR / 'cbm_resnet_feats.npy'
W2V_CACHE    = CKPT_DIR / 'cbm_w2v_feats.npy'
EGE_CACHE    = CKPT_DIR / 'ege_maps_feats.npy'
GEO_CACHE    = CKPT_DIR / 'dual_geo_scaled.npy'

# ── Teacher 체크포인트 (기존 학습된 것 로드) ──────────────────────────────────
TEACHER_CKPT = FINAL_DIR / 'kd_teacher_best.pt'

# ── 이전 Student (가중치 전이용, 없어도 됨) ───────────────────────────────────
STUDENT_CKPT = FINAL_DIR / 'kd_student_prev_best.pt'

# ── 출력 ──────────────────────────────────────────────────────────────────────
STUDENT_SAVE = FINAL_DIR / 'student_kd_best.pt'
FUSED_CACHE  = CKPT_DIR  / 'student_kd_fused_expanded.npy'
LOGITS_CACHE = CKPT_DIR  / 'student_kd_logits.npy'

PHYSIO_DIM = 132

# KD 하이퍼파라미터
ALPHA = 0.4;  BETA = 0.3;  GAMMA = 0.3;  TEMP = 4.0
S_LR  = 3e-4; S_EPOCHS = 120; S_PATIENCE = 20; S_BATCH = 32
SEED  = 42

torch.manual_seed(SEED); np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_data(samples_all):
    for p, name in [(RESNET_CACHE, 'ResNet'), (W2V_CACHE, 'Wav2Vec2'),
                    (EGE_CACHE, 'eGeMAPS'),   (GEO_CACHE, 'Geometry')]:
        if not p.exists():
            raise FileNotFoundError(f"{name} 캐시 없음: {p}")

    resnet_np = np.load(RESNET_CACHE).astype(np.float32)
    w2v_np    = np.load(W2V_CACHE).astype(np.float32)
    ege_np    = np.load(EGE_CACHE).astype(np.float32)
    geo_np    = np.load(GEO_CACHE).astype(np.float32)

    # eGeMAPS per-feature z-score
    ege_np = (ege_np - ege_np.mean(0)) / (ege_np.std(0) + 1e-6)

    # physio (Teacher KD용)
    physio_df = pd.read_csv(config.PHYSIO_FEATURES, index_col=0)
    task_keys = [s['task'] for s in samples_all]
    physio_np = np.zeros((len(samples_all), PHYSIO_DIM), dtype=np.float32)
    pmask     = np.zeros(len(samples_all), dtype=bool)
    for i, key in enumerate(task_keys):
        if key in physio_df.index:
            raw = physio_df.loc[key].values[:PHYSIO_DIM].astype(np.float32)
            if not np.any(np.isnan(raw)):
                physio_np[i] = raw; pmask[i] = True

    scaler = StandardScaler()
    physio_np[pmask] = scaler.fit_transform(physio_np[pmask]).astype(np.float32)

    labels  = np.array([s['label']   for s in samples_all], dtype=np.int64)
    subj    = np.array([s['subject'] for s in samples_all])
    task_id = np.array([
        TASK_TO_ID.get(s['task'].split('_', 1)[1] if '_' in s['task'] else s['task'], 0)
        for s in samples_all], dtype=np.int64)

    print(f"  ResNet:   {resnet_np.shape}")
    print(f"  Wav2Vec2: {w2v_np.shape}")
    print(f"  eGeMAPS:  {ege_np.shape}  (z-score 정규화 완료)")
    print(f"  Geometry: {geo_np.shape}")
    print(f"  Physio:   {physio_np.shape}  (유효 {pmask.sum()}개)")

    return {
        'resnet':   torch.tensor(resnet_np),
        'audio':    torch.tensor(w2v_np),
        'ege':      torch.tensor(ege_np),
        'geo':      torch.tensor(geo_np),
        'physio':   torch.tensor(physio_np),
        'pmask':    pmask,
        'labels':   torch.tensor(labels, dtype=torch.long),
        'task_ids': torch.tensor(task_id, dtype=torch.long),
        'subjects': subj,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Student forward 헬퍼 (eGeMAPS 포함)
# ══════════════════════════════════════════════════════════════════════════════

def ff_student(student, ff, ae, geo, ti, ege=None):
    lstm_out, _ = student.video_lstm(ff)
    g_emb       = student.geo_gru(geo)
    fused, _    = student.cross_attn(lstm_out, ae)
    fused_exp   = torch.cat([fused, g_emb], dim=1)         # (B, 320)
    if ege is not None:
        e_emb   = student.ege_branch(ege)
        fused_exp = student.ege_fusion(torch.cat([fused_exp, e_emb], dim=1))
    t_emb   = student.task_emb(ti)
    logits  = student.classifier(torch.cat([fused_exp, t_emb], dim=1))
    return logits, fused_exp


# ══════════════════════════════════════════════════════════════════════════════
#  StudentCMABi KD 학습
# ══════════════════════════════════════════════════════════════════════════════

def train_student(student, teacher, data, tr_idx, val_idx, base_acc):
    print(f"\n[KD] StudentCMABi")
    print(f"  eGeMAPS late fusion: {EGEMAP_DIM} → 64 → fused_expanded(320)")
    print(f"  α(KL)={ALPHA}  β(MSE)={BETA}  γ(CE)={GAMMA}  T={TEMP}")

    ff  = data['resnet'].to(DEVICE)
    ae  = data['audio'].to(DEVICE)
    ege = data['ege'].to(DEVICE)
    geo = data['geo'].to(DEVICE)
    ph  = data['physio'].to(DEVICE)
    lb  = data['labels'].to(DEVICE)
    ti  = data['task_ids'].to(DEVICE)
    pmask = data['pmask']

    tr_arr = torch.tensor(tr_idx)
    vi     = torch.tensor(val_idx)

    opt   = torch.optim.Adam(student.parameters(), lr=S_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, S_EPOCHS)

    def eval_val():
        student.eval()
        with torch.no_grad():
            logits, _ = ff_student(student, ff[vi], ae[vi], geo[vi], ti[vi], ege[vi])
        preds = logits.argmax(1).cpu().numpy()
        labs  = lb[vi].cpu().numpy()
        return accuracy_score(labs, preds), f1_score(labs, preds, average='weighted')

    best_acc, best_state, patience = 0.0, None, 0

    for ep in range(1, S_EPOCHS + 1):
        student.train(); teacher.eval()
        perm = torch.randperm(len(tr_arr))
        tr_correct, tr_n = 0, 0

        for b in range(0, len(perm), S_BATCH):
            bi = tr_arr[perm[b:b+S_BATCH]]
            has_p = torch.tensor(pmask)[bi].to(DEVICE)

            with torch.no_grad():
                t_logits, t_fused = teacher(ff[bi], ae[bi], ph[bi], geo[bi], ti[bi])

            s_logits, s_fused = ff_student(student, ff[bi], ae[bi], geo[bi], ti[bi], ege[bi])

            L_ce = F.cross_entropy(s_logits, lb[bi])
            if has_p.any():
                L_kl = F.kl_div(
                    F.log_softmax(s_logits[has_p] / TEMP, dim=1),
                    F.softmax(t_logits[has_p].detach() / TEMP, dim=1),
                    reduction='batchmean') * (TEMP ** 2)
                L_mse = F.mse_loss(s_fused[has_p], t_fused[has_p].detach())
            else:
                L_kl  = torch.tensor(0.0, device=DEVICE)
                L_mse = torch.tensor(0.0, device=DEVICE)

            loss = GAMMA * L_ce + ALPHA * L_kl + BETA * L_mse

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            tr_correct += (s_logits.argmax(1) == lb[bi]).sum().item()
            tr_n += len(bi)
        sched.step()

        val_acc, val_f1 = eval_val()
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
            patience   = 0; mark = " ★"
        else:
            patience += 1; mark = ""
            if patience >= S_PATIENCE:
                print(f"    Early stop EP{ep}  best={best_acc:.4f}"); break

        if ep % 10 == 0 or ep <= 3:
            print(f"  EP{ep:3d}  tr={tr_correct/tr_n:.4f}"
                  f"  val={val_acc:.4f}  f1={val_f1:.4f}{mark}")

    student.load_state_dict(best_state)
    final_acc, final_f1 = eval_val()

    # fused_expanded 전체 캐시 저장
    student.eval()
    N = len(data['labels'])
    fused_all  = np.zeros((N, FUSED_EXP_DIM), dtype=np.float32)
    logits_all = np.zeros((N, 2), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, N, 64):
            e  = min(b + 64, N)
            bi = torch.arange(b, e, device=DEVICE)
            lg, fe = ff_student(student, ff[bi], ae[bi], geo[bi], ti[bi], ege[bi])
            fused_all[b:e]  = fe.cpu().numpy()
            logits_all[b:e] = lg.cpu().numpy()
    np.save(FUSED_CACHE,  fused_all)
    np.save(LOGITS_CACHE, logits_all)

    print(f"\n{'='*60}")
    print(f"  KD 학습 결과")
    print(f"  Acc: {base_acc:.4f}  →  {final_acc:.4f}  (Δ{final_acc-base_acc:+.4f})")
    print(f"  F1:  —         →  {final_f1:.4f}")
    print(f"{'='*60}")

    torch.save({
        'model_state_dict': best_state,
        'val_acc': final_acc, 'val_f1': final_f1,
        'egemap_dim': EGEMAP_DIM,
        'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA, 'temp': TEMP,
    }, STUDENT_SAVE)
    print(f"✅ 저장: {STUDENT_SAVE}")
    return final_acc


# ══════════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  StudentCMABi KD 학습")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    if not EGE_CACHE.exists():
        print(f"\n⛔  eGeMAPS 캐시 없음: {EGE_CACHE}")
        print("    먼저 실행: python ai/feature_analysis/extract_ege_maps.py")
        return

    if not TEACHER_CKPT.exists():
        print(f"\n⛔  Teacher 체크포인트 없음: {TEACHER_CKPT}")
        return

    labels_df  = pd.read_csv(config.LABELS_FILE)
    labels_map = dict(zip(labels_df.iloc[:, 0], labels_df.iloc[:, 1]))
    samples    = build_sample_list(labels_map)

    print("\n[데이터 로드]")
    data = load_data(samples)

    labels   = data['labels'].numpy()
    subjects = data['subjects']
    gss = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(gss.split(range(len(samples)), labels, subjects))
    print(f"\n  Train: {len(tr_idx)}  Val: {len(val_idx)}")

    # ── Teacher 로드 ──────────────────────────────────────────────────────────
    print(f"\n[Teacher 로드]  {TEACHER_CKPT.name}")
    teacher = Teacher(physio_in=PHYSIO_DIM).to(DEVICE)
    raw = torch.load(TEACHER_CKPT, map_location=DEVICE)
    raw_state = raw.get('model_state_dict', raw)
    # 구 아키텍처 키 호환 (geo_gru.pool.score.* → geo_gru.attn_score.*)
    remapped = {k.replace('geo_gru.pool.score.', 'geo_gru.attn_score.'): v
                for k, v in raw_state.items()}
    teacher.load_state_dict(remapped, strict=False)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    vi = torch.tensor(val_idx)
    with torch.no_grad():
        t_logits, _ = teacher(
            data['resnet'][vi].to(DEVICE),
            data['audio'][vi].to(DEVICE),
            data['physio'][vi].to(DEVICE),
            data['geo'][vi].to(DEVICE),
            data['task_ids'][vi].to(DEVICE),
        )
    t_acc = accuracy_score(labels[val_idx], t_logits.argmax(1).cpu().numpy())
    print(f"  Teacher val_acc: {t_acc:.4f}")

    # ── StudentCMABi 초기화 + 가중치 전이 ────────────────────────────────────
    print("\n[Student 초기화]")
    student = StudentCMABi(dropout=0.3).to(DEVICE)
    if STUDENT_CKPT.exists():
        prev_state = torch.load(STUDENT_CKPT, map_location=DEVICE)['model_state_dict']
        s_state    = student.state_dict()
        transfer   = {k: v for k, v in prev_state.items()
                      if k in s_state and s_state[k].shape == v.shape}
        s_state.update(transfer)
        student.load_state_dict(s_state)
        print(f"  가중치 전이: {len(transfer)}개")
    else:
        print("  랜덤 초기화")

    student.eval()
    with torch.no_grad():
        lg, _ = ff_student(
            student,
            data['resnet'][vi].to(DEVICE),
            data['audio'][vi].to(DEVICE),
            data['geo'][vi].to(DEVICE),
            data['task_ids'][vi].to(DEVICE),
        )
    base_acc = accuracy_score(labels[val_idx], lg.argmax(1).cpu().numpy())
    print(f"  Baseline (eGeMAPS 없음): {base_acc:.4f}")

    student_acc = train_student(student, teacher, data, list(tr_idx), list(val_idx), base_acc)

    print(f"\n{'='*60}")
    print(f"  최종 결과")
    print(f"  Teacher : {t_acc:.4f}")
    print(f"  Student : {student_acc:.4f}")
    print(f"  기존 best: 0.8099")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
