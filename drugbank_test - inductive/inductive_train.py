from datetime import datetime
from pathlib import Path
import argparse, random, time, warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch import optim

import models
import custom_loss
from data_preprocessing import (
    DrugDataset, DrugDataLoader, configure_ddi_statistics,
    load_fg_enrichment_scores, precompute_functional_groups,
)

warnings.filterwarnings('ignore', category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55)
parser.add_argument('--n_atom_hid', type=int, default=128)
parser.add_argument('--rel_total', type=int, default=86)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--kge_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=float, default=1.0)
parser.add_argument('--use_cuda', type=int, default=1, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='inductive.pkl')
parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2])
parser.add_argument('--seed', type=int, default=0)

# Validation-only early stopping.
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--min_delta', type=float, default=0.001)

# Same Stage04 CL definition/defaults as frozen Full SGAR.
parser.add_argument('--cl_weight', type=float, default=0.05)
parser.add_argument('--cl_mask_ratio', type=float, default=0.10)
parser.add_argument('--cl_temperature', type=float, default=0.20)

args = parser.parse_args()

CL_WEIGHT = args.cl_weight
CL_MASK_RATIO = args.cl_mask_ratio
CL_TEMPERATURE = args.cl_temperature

# ============================================================
# Reproducibility
# ============================================================

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = (
    'cuda:0'
    if torch.cuda.is_available() and args.use_cuda
    else 'cpu'
)

pkl_name = args.pkl_name

if pkl_name == 'inductive.pkl':
    pkl_name = (
        f'inductive_drugbank_'
        f'fold{args.fold}_seed{args.seed}.pkl'
    )

# ============================================================
# Frozen clean protocol
# ============================================================

protocol_dir = (
    Path(__file__).resolve().parents[1]
    / 'protocol'
    / 'drugbank_inductive'
    / f'fold{args.fold}'
)


def to_tuples(df):
    return [
        (str(h), str(t), int(r))
        for h, t, r in zip(
            df.d1,
            df.d2,
            df.type,
        )
    ]


train_tup = to_tuples(
    pd.read_csv(
        protocol_dir / 'train.csv'
    )
)

val_tup = to_tuples(
    pd.read_csv(
        protocol_dir / 'val.csv'
    )
)

s1_tup = to_tuples(
    pd.read_csv(
        protocol_dir / 's1.csv'
    )
)

s2_tup = to_tuples(
    pd.read_csv(
        protocol_dir / 's2.csv'
    )
)

# ============================================================
# TRAIN-ONLY statistics
# ============================================================

ddi_info = configure_ddi_statistics(
    train_tup
)

fg_info = load_fg_enrichment_scores(
    protocol_dir / 'fg_statistics.csv'
)

precompute_functional_groups()

print(args)
print(f'[device] {device}')

print(
    f'[clean fold] '
    f'train={len(train_tup)} '
    f'val={len(val_tup)} '
    f'S1={len(s1_tup)} '
    f'S2={len(s2_tup)}'
)

print(
    f"[train-only DDI] "
    f"triples={ddi_info['num_triples']} "
    f"drugs={ddi_info['num_drugs']} "
    f"relations={ddi_info['num_relations']}"
)

print(
    f"[train-only FG] "
    f"rows={fg_info['rows']} "
    f"min/median/max="
    f"{fg_info['min_enrichment']:.6f}/"
    f"{fg_info['median_enrichment']:.6f}/"
    f"{fg_info['max_enrichment']:.6f}"
)

print(
    f'[Stage04-CL] '
    f'weight={CL_WEIGHT} '
    f'mask_ratio={CL_MASK_RATIO} '
    f'temperature={CL_TEMPERATURE}'
)

# ============================================================
# Dataset
#
# train : dynamic negatives using train-only statistics
# val   : frozen negatives
# S1    : frozen negatives
# S2    : frozen negatives
# ============================================================

train_data = DrugDataset(
    train_tup,
    ratio=args.data_size_ratio,
    neg_ent=args.neg_samples,
    disjoint_split=True,
    shuffle=True,
)

val_data = DrugDataset(
    val_tup,
    disjoint_split=True,
    shuffle=False,
    fixed_negative_file=(
        protocol_dir / 'val_negatives.csv'
    ),
)

s1_data = DrugDataset(
    s1_tup,
    disjoint_split=True,
    shuffle=False,
    fixed_negative_file=(
        protocol_dir / 's1_negatives.csv'
    ),
)

s2_data = DrugDataset(
    s2_tup,
    disjoint_split=True,
    shuffle=False,
    fixed_negative_file=(
        protocol_dir / 's2_negatives.csv'
    ),
)


def seed_worker(_):
    worker_seed = (
        torch.initial_seed()
        % (2 ** 32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generator(offset):
    g = torch.Generator()

    g.manual_seed(
        args.seed + offset
    )

    return g


train_loader = DrugDataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=generator(11),
)

val_loader = DrugDataLoader(
    val_data,
    batch_size=args.batch_size * 3,
    shuffle=False,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=generator(22),
)

s1_loader = DrugDataLoader(
    s1_data,
    batch_size=args.batch_size * 3,
    shuffle=False,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=generator(33),
)

s2_loader = DrugDataLoader(
    s2_data,
    batch_size=args.batch_size * 3,
    shuffle=False,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=generator(44),
)

# ============================================================
# Stage04 CL
# ============================================================


def info_nce(z1, z2):

    z1 = F.normalize(
        z1,
        dim=-1,
    )

    z2 = F.normalize(
        z2,
        dim=-1,
    )

    logits = (
        z1 @ z2.T
        / CL_TEMPERATURE
    )

    labels = torch.arange(
        z1.size(0),
        device=z1.device,
    )

    return 0.5 * (
        F.cross_entropy(
            logits,
            labels,
        )
        +
        F.cross_entropy(
            logits.T,
            labels,
        )
    )


def mask_x(x):

    out = x.detach().clone()

    if (
        out.numel() == 0
        or out.size(0) == 0
        or CL_MASK_RATIO <= 0
    ):
        return out

    n = min(
        max(
            1,
            int(
                out.size(0)
                * CL_MASK_RATIO
            ),
        ),
        out.size(0),
    )

    idx = torch.randperm(
        out.size(0),
        device=out.device,
    )[:n]

    out[idx] = 0.0

    return out


def compute_cl(
    model,
    raw,
    h_base,
    t_base,
):

    h, t, r, b = raw

    h_aug = h.clone()
    t_aug = t.clone()
    b_aug = b.clone()

    h_aug.x = mask_x(
        h_aug.x
    )

    t_aug.x = mask_x(
        t_aug.x
    )

    (
        _,
        h_last_aug,
        t_last_aug,
    ) = model(
        (
            h_aug,
            t_aug,
            r,
            b_aug,
        ),
        return_last_repr=True,
    )

    loss_h = info_nce(
        model.cl_proj(h_base),
        model.cl_proj(h_last_aug),
    )

    loss_t = info_nce(
        model.cl_proj(t_base),
        model.cl_proj(t_last_aug),
    )

    return 0.5 * (
        loss_h + loss_t
    )


def do_compute(
    batch,
    model,
    use_cl=False,
):

    pos, neg = batch

    pos = [
        x.to(device)
        for x in pos
    ]

    if use_cl:

        # Keep untouched copies because
        # the model mutates x internally.
        raw = (
            pos[0].clone(),
            pos[1].clone(),
            pos[2],
            pos[3].clone(),
        )

        (
            p_score,
            h_base,
            t_base,
        ) = model(
            pos,
            return_last_repr=True,
        )

        cl_loss = compute_cl(
            model,
            raw,
            h_base,
            t_base,
        )

    else:

        p_score = model(pos)

        cl_loss = torch.zeros(
            (),
            device=device,
        )

    neg = [
        x.to(device)
        for x in neg
    ]

    n_score = model(neg)

    prob = np.concatenate([
        torch.sigmoid(
            p_score.detach()
        ).cpu().numpy(),

        torch.sigmoid(
            n_score.detach()
        ).cpu().numpy(),
    ])

    target = np.concatenate([
        np.ones(
            len(p_score)
        ),
        np.zeros(
            len(n_score)
        ),
    ])

    rels = np.concatenate([
        pos[2]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1),

        neg[2]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1),
    ])

    return (
        p_score,
        n_score,
        prob,
        target,
        rels,
        cl_loss,
    )

# ============================================================
# Metrics
# ============================================================


def metric_dict(
    prob,
    target,
):

    pred = (
        prob >= 0.5
    ).astype(int)

    return {
        'acc':
            metrics.accuracy_score(
                target,
                pred,
            ),

        'auroc':
            metrics.roc_auc_score(
                target,
                prob,
            ),

        'f1':
            metrics.f1_score(
                target,
                pred,
            ),

        'precision':
            metrics.precision_score(
                target,
                pred,
                zero_division=0,
            ),

        'recall':
            metrics.recall_score(
                target,
                pred,
                zero_division=0,
            ),

        'ap':
            metrics.average_precision_score(
                target,
                prob,
            ),
    }


def evaluate(
    loader,
    model,
    loss_fn,
    per_rel=False,
):

    model.eval()

    total_loss = 0.0

    probs = []
    targets = []
    rels_all = []

    with torch.no_grad():

        for batch in loader:

            (
                p,
                n,
                prob,
                target,
                rels,
                _,
            ) = do_compute(
                batch,
                model,
                use_cl=False,
            )

            batch_loss, _, _ = (
                loss_fn(
                    p,
                    n,
                )
            )

            total_loss += (
                batch_loss.item()
                * len(p)
            )

            probs.append(prob)
            targets.append(target)
            rels_all.append(rels)

    prob = np.concatenate(
        probs
    )

    target = np.concatenate(
        targets
    )

    rels = np.concatenate(
        rels_all
    )

    out = metric_dict(
        prob,
        target,
    )

    out['loss'] = (
        total_loss
        / len(loader.dataset)
    )

    if per_rel:

        per = {}

        for rel in np.unique(
            rels
        ):

            mask = (
                rels == rel
            )

            if (
                target[mask].sum() == 0
                or
                (
                    1
                    - target[mask]
                ).sum() == 0
            ):

                per[int(rel)] = {
                    'error':
                        'Insufficient samples '
                        'for both classes'
                }

            else:

                per[int(rel)] = (
                    metric_dict(
                        prob[mask],
                        target[mask],
                    )
                )

        out['per_rel'] = per

    return out

# ============================================================
# Training
#
# CRITICAL:
# S1 and S2 NEVER enter this function.
# ============================================================


def train(
    model,
    loss_fn,
    optimizer,
    scheduler,
):

    print(
        'Starting training at',
        datetime.today(),
    )

    best_acc = float('-inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(
        1,
        args.n_epochs + 1,
    ):

        start = time.time()

        total_sum = 0.0
        ddi_sum = 0.0
        cl_sum = 0.0

        probs = []
        targets = []

        for batch in train_loader:

            model.train()

            (
                p,
                n,
                prob,
                target,
                _,
                cl_loss,
            ) = do_compute(
                batch,
                model,
                use_cl=True,
            )

            ddi_loss, _, _ = (
                loss_fn(
                    p,
                    n,
                )
            )

            total_loss = (
                ddi_loss
                +
                CL_WEIGHT
                * cl_loss
            )

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            k = len(p)

            total_sum += (
                total_loss.item()
                * k
            )

            ddi_sum += (
                ddi_loss.item()
                * k
            )

            cl_sum += (
                cl_loss.item()
                * k
            )

            probs.append(prob)
            targets.append(target)

        train_m = metric_dict(
            np.concatenate(probs),
            np.concatenate(targets),
        )

        # DDI-only validation.
        val_m = evaluate(
            val_loader,
            model,
            loss_fn,
            per_rel=False,
        )

        # Keep the same selection metric as
        # frozen clean transductive protocol:
        # validation ACC.
        improved = (
            val_m['acc']
            >
            best_acc
            + args.min_delta
        )

        if improved:

            best_acc = (
                val_m['acc']
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                model,
                pkl_name,
            )

            print(
                f'*** best VAL checkpoint '
                f'epoch={epoch} '
                f'ACC={best_acc:.6f} ***'
            )

        else:

            patience_counter += 1

        if scheduler:
            scheduler.step()

        n_train = len(
            train_loader.dataset
        )

        print(
            f'Epoch {epoch} '
            f'({time.time()-start:.2f}s) '
            f'total={total_sum/n_train:.6f} '
            f'ddi={ddi_sum/n_train:.6f} '
            f'cl={cl_sum/n_train:.6f} '
            f'val_loss={val_m["loss"]:.6f}'
        )

        print(
            f'  train '
            f'ACC/AUROC/AP='
            f'{train_m["acc"]:.6f}/'
            f'{train_m["auroc"]:.6f}/'
            f'{train_m["ap"]:.6f}'
        )

        print(
            f'  val   '
            f'ACC/AUROC/AP='
            f'{val_m["acc"]:.6f}/'
            f'{val_m["auroc"]:.6f}/'
            f'{val_m["ap"]:.6f} '
            f'patience='
            f'{patience_counter}/'
            f'{args.patience}'
        )

        if (
            patience_counter
            >= args.patience
        ):

            print(
                f'Early stopping at '
                f'epoch {epoch}; '
                f'best epoch='
                f'{best_epoch}, '
                f'val ACC='
                f'{best_acc:.6f}'
            )

            break

    return (
        best_epoch,
        best_acc,
    )


def save_results(
    s1,
    s2,
):

    rows = []

    for name, result in [
        ('S1', s1),
        ('S2', s2),
    ]:

        rows.append({
            'Dataset':
                name,

            'Relation':
                'Overall',

            'ACC':
                result['acc'],

            'AUROC':
                result['auroc'],

            'AP':
                result['ap'],

            'F1':
                result['f1'],

            'Precision':
                result['precision'],

            'Recall':
                result['recall'],
        })

        for rel, value in (
            result['per_rel'].items()
        ):

            row = {
                'Dataset':
                    name,

                'Relation':
                    rel,
            }

            if (
                'error'
                in value
            ):

                row['Error'] = (
                    value['error']
                )

            else:

                row.update({
                    'ACC':
                        value['acc'],

                    'AUROC':
                        value['auroc'],

                    'AP':
                        value['ap'],

                    'F1':
                        value['f1'],

                    'Precision':
                        value['precision'],

                    'Recall':
                        value['recall'],
                })

            rows.append(row)

    out = pd.DataFrame(
        rows
    )

    out_file = (
        f'inductive_drugbank_'
        f'fold{args.fold}_'
        f'seed{args.seed}_'
        f'per_relation_metrics.csv'
    )

    out.to_csv(
        out_file,
        index=False,
    )

    print(
        f'Saved {out_file}'
    )


# ============================================================
# Main
# ============================================================

model = models.MVN_DDI(
    args.n_atom_feats,
    args.n_atom_hid,
    args.kge_dim,
    args.rel_total,
    heads_out_feat_params=[
        64,
        64,
        64,
        64,
    ],
    blocks_params=[
        2,
        2,
        2,
        2,
    ],
).to(device)

loss_fn = (
    custom_loss.SigmoidLoss()
)

optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

scheduler = (
    optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch:
            0.96 ** epoch,
    )
)

best_epoch, best_val_acc = (
    train(
        model,
        loss_fn,
        optimizer,
        scheduler,
    )
)

print(
    '\nLoading ONE checkpoint '
    'selected only by validation...'
)

best_model = torch.load(
    pkl_name,
    map_location=device,
)

best_model.to(device)

# ============================================================
# FINAL TEST
#
# S1/S2 first appear only after
# checkpoint selection has completed.
# ============================================================

s1_result = evaluate(
    s1_loader,
    best_model,
    loss_fn,
    per_rel=True,
)

s2_result = evaluate(
    s2_loader,
    best_model,
    loss_fn,
    per_rel=True,
)

print()
print(
    '===== FINAL INDUCTIVE TEST ====='
)

print(
    f'best_epoch={best_epoch} '
    f'best_val_ACC='
    f'{best_val_acc:.6f}'
)

for name, result in [
    ('S1', s1_result),
    ('S2', s2_result),
]:

    print(
        f'{name} '
        f'ACC/AUROC/AP/F1/'
        f'Precision/Recall='
        f'{result["acc"]:.6f}/'
        f'{result["auroc"]:.6f}/'
        f'{result["ap"]:.6f}/'
        f'{result["f1"]:.6f}/'
        f'{result["precision"]:.6f}/'
        f'{result["recall"]:.6f}'
    )

save_results(
    s1_result,
    s2_result,
)
