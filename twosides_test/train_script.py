from datetime import datetime
from pathlib import Path
import time
import argparse
import torch
import torch.nn.functional as F
import warnings
import random

from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np

import models
import custom_loss
from data_preprocessing import (
    DrugDataset,
    DrugDataLoader,
    load_fg_enrichment_scores,
    precompute_functional_groups,
    configure_ddi_statistics,
)

warnings.filterwarnings('ignore', category=UserWarning)

# =========================================================
# Parameters: preserve the official TWOSIDES optimization
# defaults while using the frozen clean TWOSIDES protocol.
# Stage 04 adds auxiliary contrastive learning.
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55)
parser.add_argument('--n_atom_hid', type=int, default=128)
parser.add_argument('--rel_total', type=int, default=963)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--kge_dim', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='transductive_twosides.pkl')
parser.add_argument('--repeat', type=int, default=0, choices=[0, 1, 2])
parser.add_argument('--seed', type=int, default=0)

# Stage-04 CL defaults reproduce the historical SGAR-DDI settings.
parser.add_argument('--cl_weight', type=float, default=0.05)
parser.add_argument('--cl_mask_ratio', type=float, default=0.10)
parser.add_argument('--cl_temperature', type=float, default=0.20)

args = parser.parse_args()

n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
pkl_name = args.pkl_name
weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
repeat = args.repeat
seed = args.seed
CL_WEIGHT = args.cl_weight
CL_MASK_RATIO = args.cl_mask_ratio
CL_TEMPERATURE = args.cl_temperature

# =========================================================
# Reproducibility
# =========================================================
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Avoid overwriting checkpoints across repeats/seeds.
if pkl_name == 'transductive_twosides.pkl':
    pkl_name = (
        f'transductive_twosides_'
        f'repeat{repeat}_seed{seed}.pkl'
    )

device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
print(
    f'[Stage04-CL] weight={CL_WEIGHT}, mask_ratio={CL_MASK_RATIO}, '
    f'temperature={CL_TEMPERATURE}'
)

# =========================================================
# Frozen clean protocol path
# =========================================================
protocol_dir = (
    Path(__file__).resolve().parents[1]
    / "protocol"
    / "twosides"
    / f"repeat{repeat}"
)

# =========================================================
# Train-only functional-group setup
# =========================================================
print('Loading train-only functional group enrichment scores...')

fg_info = load_fg_enrichment_scores(
    protocol_dir / "fg_statistics.csv"
)

print(
    f"[Train-only FG] rows={fg_info['rows']}, "
    f"min={fg_info['min_enrichment']:.6f}, "
    f"median={fg_info['median_enrichment']:.6f}, "
    f"max={fg_info['max_enrichment']:.6f}"
)

print('Precomputing functional groups for all molecules...')
precompute_functional_groups()
print('Functional group setup complete.')

# =========================================================
# Dataset: frozen clean TWOSIDES pair-disjoint splits
# =========================================================

df_ddi_train = pd.read_csv(protocol_dir / "train.csv")
df_ddi_val = pd.read_csv(protocol_dir / "val.csv")
df_ddi_test = pd.read_csv(protocol_dir / "test.csv")


def dataframe_to_tuples(df):
    return [
        (h, t, int(r))
        for h, t, r in zip(
            df["d1"],
            df["d2"],
            df["type"],
        )
    ]


train_tup = dataframe_to_tuples(df_ddi_train)
val_tup = dataframe_to_tuples(df_ddi_val)
test_tup = dataframe_to_tuples(df_ddi_test)

# IMPORTANT:
# negative-sampling statistics are built from TRAIN only.
ddi_stats = configure_ddi_statistics(train_tup)

print(
    f"[Clean protocol] repeat={repeat}, "
    f"train={len(train_tup)}, "
    f"val={len(val_tup)}, "
    f"test={len(test_tup)}"
)

print(
    f"[Train-only DDI statistics] "
    f"triples={ddi_stats['num_triples']}, "
    f"relations={ddi_stats['num_relations']}"
)

train_data = DrugDataset(
    train_tup,
    ratio=data_size_ratio,
    neg_ent=neg_samples,
)

# Frozen validation positives + deterministic fixed negatives.
val_data = DrugDataset(
    val_tup,
    ratio=data_size_ratio,
    disjoint_split=False,
    shuffle=False,
    fixed_negative_file=(
        protocol_dir
        / "val_negatives.csv.gz"
    ),
)

# Frozen test positives + deterministic fixed negatives.
test_data = DrugDataset(
    test_tup,
    ratio=data_size_ratio,
    disjoint_split=False,
    shuffle=False,
    fixed_negative_file=(
        protocol_dir
        / "test_negatives.csv.gz"
    ),
)

print(
    f'Training with {len(train_data)} samples, '
    f'validating with {len(val_data)}, and testing with {len(test_data)}'
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_generator = torch.Generator()
train_generator.manual_seed(seed + 1000)

val_generator = torch.Generator()
val_generator.manual_seed(seed + 2000)

test_generator = torch.Generator()
test_generator.manual_seed(seed + 3000)


train_data_loader = DrugDataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=train_generator,
)
val_data_loader = DrugDataLoader(
    val_data,
    batch_size=batch_size * 3,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=val_generator,
)
test_data_loader = DrugDataLoader(
    test_data,
    batch_size=batch_size * 3,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=test_generator,
)

# =========================================================
# Stage 04: auxiliary contrastive learning
# =========================================================
def _info_nce(z1, z2, temperature=CL_TEMPERATURE):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.transpose(0, 1)) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_12 + loss_21)


def _mask_x_tensor(x, mask_ratio=CL_MASK_RATIO):
    out = x.detach().clone()
    if out.numel() == 0 or out.size(0) == 0 or mask_ratio <= 0:
        return out

    num_nodes = out.size(0)
    num_mask = max(1, int(num_nodes * mask_ratio))
    num_mask = min(num_mask, num_nodes)
    perm = torch.randperm(num_nodes, device=out.device)[:num_mask]
    out[perm] = 0.0
    return out


def _compute_cl_loss(
    model,
    pos_tri_raw,
    h_last_base,
    t_last_base,
    mask_ratio=CL_MASK_RATIO,
    temperature=CL_TEMPERATURE,
):
    """
    Build one masked positive view from the original positive drug pair.
    The DDI relation and graph topology are unchanged; only atom features
    are masked. Both the original and augmented views remain connected to
    the encoder, so the CL loss regularizes the representation backbone as
    well as the projection head.
    """
    h_data_raw, t_data_raw, rels, b_graph_raw = pos_tri_raw

    h_data_aug = h_data_raw.clone()
    t_data_aug = t_data_raw.clone()
    b_graph_aug = b_graph_raw.clone()

    h_data_aug.x = _mask_x_tensor(h_data_aug.x, mask_ratio)
    t_data_aug.x = _mask_x_tensor(t_data_aug.x, mask_ratio)

    _, h_last_aug, t_last_aug = model(
        (h_data_aug, t_data_aug, rels, b_graph_aug),
        return_last_repr=True,
    )

    h_proj = model.cl_proj(h_last_base)
    t_proj = model.cl_proj(t_last_base)
    h_proj_aug = model.cl_proj(h_last_aug)
    t_proj_aug = model.cl_proj(t_last_aug)

    loss_h = _info_nce(h_proj, h_proj_aug, temperature)
    loss_t = _info_nce(t_proj, t_proj_aug, temperature)
    return 0.5 * (loss_h + loss_t)


def do_compute(batch, device, model, compute_cl=False):
    """
    batch: (pos_tri, neg_tri)
    pos/neg_tri: (batch_h, batch_t, batch_r, b_graph)
    """
    probas_pred, ground_truth, rel_types = [], [], []
    pos_tri, neg_tri = batch

    pos_tri = [obj.to(device=device) for obj in pos_tri]

    if compute_cl:
        # Preserve the untouched positive graph before the main forward mutates x.
        pos_tri_raw = (
            pos_tri[0].clone(),
            pos_tri[1].clone(),
            pos_tri[2],
            pos_tri[3].clone(),
        )
        p_score, h_last_base, t_last_base = model(
            pos_tri,
            return_last_repr=True,
        )
        cl_loss = _compute_cl_loss(
            model,
            pos_tri_raw,
            h_last_base,
            t_last_base,
        )
    else:
        p_score = model(pos_tri)
        cl_loss = torch.zeros((), device=device)

    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))
    rel_types.append(pos_tri[2].squeeze().cpu().numpy())

    neg_tri = [obj.to(device=device) for obj in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))
    rel_types.append(neg_tri[2].squeeze().cpu().numpy())

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)
    rel_types = np.concatenate(rel_types)

    if compute_cl:
        return p_score, n_score, probas_pred, ground_truth, rel_types, cl_loss
    return p_score, n_score, probas_pred, ground_truth, rel_types


# =========================================================
# Metrics: unchanged baseline definitions
# =========================================================
def do_compute_metrics(probas_pred, target, rel_types=None, per_rel=False):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, _ = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)

    if per_rel and rel_types is not None:
        rel_metrics = {}
        unique_rels = np.unique(rel_types)
        target_array = np.array(target)

        for rel in unique_rels:
            rel_mask = rel_types == rel
            if (
                sum(rel_mask) > 0
                and sum(target_array[rel_mask]) > 0
                and sum(1 - target_array[rel_mask]) > 0
            ):
                try:
                    rel_pred = pred[rel_mask]
                    rel_target = target_array[rel_mask]
                    rel_probas = probas_pred[rel_mask]
                    rel_p, rel_r, _ = metrics.precision_recall_curve(
                        rel_target,
                        rel_probas,
                    )
                    rel_metrics[rel] = {
                        'acc': metrics.accuracy_score(rel_target, rel_pred),
                        'auroc': metrics.roc_auc_score(rel_target, rel_probas),
                        'f1': metrics.f1_score(rel_target, rel_pred),
                        'precision': metrics.precision_score(rel_target, rel_pred),
                        'recall': metrics.recall_score(rel_target, rel_pred),
                        'int_ap': metrics.auc(rel_r, rel_p),
                        'ap': metrics.average_precision_score(rel_target, rel_probas),
                    }
                except Exception as exc:
                    rel_metrics[rel] = {'error': str(exc)}
            else:
                rel_metrics[rel] = {
                    'error': 'Insufficient samples for both classes'
                }

        return (
            acc,
            auroc,
            f1_score,
            precision,
            recall,
            int_ap,
            ap,
            rel_metrics,
        )

    return acc, auroc, f1_score, precision, recall, int_ap, ap


# =========================================================
# Train / validation
# =========================================================
def train(
    model,
    train_data_loader,
    val_data_loader,
    loss_fn,
    optimizer,
    n_epochs,
    device,
    scheduler=None,
):
    max_acc = 0.0
    print('Starting training at', datetime.today())

    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0.0
        train_ddi_loss = 0.0
        train_cl_loss = 0.0
        val_loss = 0.0
        train_probas_pred = []
        train_ground_truth = []
        train_rel_types = []
        val_probas_pred = []
        val_ground_truth = []
        val_rel_types = []

        for batch in train_data_loader:
            model.train()
            (
                p_score,
                n_score,
                probas_pred,
                ground_truth,
                rel_types,
                cl_loss,
            ) = do_compute(batch, device, model, compute_cl=True)

            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            train_rel_types.append(rel_types)

            ddi_loss, _, _ = loss_fn(p_score, n_score)
            loss = ddi_loss + CL_WEIGHT * cl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_pos = len(p_score)
            train_loss += loss.item() * n_pos
            train_ddi_loss += ddi_loss.item() * n_pos
            train_cl_loss += cl_loss.item() * n_pos

        train_loss /= len(train_data)
        train_ddi_loss /= len(train_data)
        train_cl_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)
            train_rel_types = np.concatenate(train_rel_types)
            (
                train_acc,
                train_auc_roc,
                train_f1,
                train_precision,
                train_recall,
                train_int_ap,
                train_ap,
            ) = do_compute_metrics(train_probas_pred, train_ground_truth)

            # Validation is intentionally DDI-only: CL is an auxiliary training loss.
            for batch in val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth, rel_types = do_compute(
                    batch,
                    device,
                    model,
                    compute_cl=False,
                )
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                val_rel_types.append(rel_types)
                ddi_val_loss, _, _ = loss_fn(p_score, n_score)
                val_loss += ddi_val_loss.item() * len(p_score)

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_rel_types = np.concatenate(val_rel_types)
            (
                val_acc,
                val_auc_roc,
                val_f1,
                val_precision,
                val_recall,
                val_int_ap,
                val_ap,
            ) = do_compute_metrics(val_probas_pred, val_ground_truth)

            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(model, pkl_name)

        if scheduler:
            scheduler.step()

        print(
            f'Epoch: {i} ({time.time() - start:.4f}s), '
            f'train_total_loss: {train_loss:.4f}, '
            f'train_ddi_loss: {train_ddi_loss:.4f}, '
            f'train_cl_loss: {train_cl_loss:.4f}, '
            f'val_loss: {val_loss:.4f}, '
            f'train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}'
        )
        print(
            f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, '
            f'train_precision: {train_precision:.4f}, '
            f'val_precision: {val_precision:.4f}'
        )


# =========================================================
# Test: no CL branch is evaluated
# =========================================================
def test(test_data_loader, model):
    test_probas_pred = []
    test_ground_truth = []
    test_rel_types = []

    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth, rel_types = do_compute(
                batch,
                device,
                model,
                compute_cl=False,
            )
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
            test_rel_types.append(rel_types)

        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_rel_types = np.concatenate(test_rel_types)

        test_metrics = do_compute_metrics(
            test_probas_pred,
            test_ground_truth,
        )
        (
            test_acc,
            test_auc_roc,
            test_f1,
            test_precision,
            test_recall,
            test_int_ap,
            test_ap,
        ) = test_metrics

        _, _, _, _, _, _, _, rel_metrics = do_compute_metrics(
            test_probas_pred,
            test_ground_truth,
            test_rel_types,
            per_rel=True,
        )

    print('\n')
    print('============================== Test Result ==============================')
    print(
        f'\t\ttest_acc: {test_acc:.4f}, '
        f'test_auc_roc: {test_auc_roc:.4f}, '
        f'test_f1: {test_f1:.4f}, '
        f'test_precision: {test_precision:.4f}'
    )
    print(
        f'\t\ttest_recall: {test_recall:.4f}, '
        f'test_int_ap: {test_int_ap:.4f}, '
        f'test_ap: {test_ap:.4f}'
    )

    adr_data = [{
        'ADR_Type': 'Overall',
        'Accuracy': test_acc,
        'AUROC': test_auc_roc,
        'F1': test_f1,
        'Precision': test_precision,
        'Recall': test_recall,
        'Int_AP': test_int_ap,
        'AP': test_ap,
    }]

    for rel, metrics_dict in rel_metrics.items():
        if isinstance(metrics_dict, dict) and 'error' not in metrics_dict:
            adr_data.append({
                'ADR_Type': rel,
                'Accuracy': metrics_dict['acc'],
                'AUROC': metrics_dict['auroc'],
                'F1': metrics_dict['f1'],
                'Precision': metrics_dict['precision'],
                'Recall': metrics_dict['recall'],
                'Int_AP': metrics_dict['int_ap'],
                'AP': metrics_dict['ap'],
            })
        else:
            error_msg = (
                metrics_dict.get('error', 'Unknown error')
                if isinstance(metrics_dict, dict)
                else 'Unknown error'
            )
            adr_data.append({
                'ADR_Type': rel,
                'Accuracy': None,
                'AUROC': None,
                'F1': None,
                'Precision': None,
                'Recall': None,
                'Int_AP': None,
                'AP': None,
                'Error': error_msg,
            })

    adr_df = pd.DataFrame(adr_data)
    overall_row = adr_df[adr_df['ADR_Type'] == 'Overall']
    per_adr_rows = adr_df[adr_df['ADR_Type'] != 'Overall'].sort_values(
        by=['Accuracy', 'ADR_Type'],
        ascending=[False, True],
    )
    adr_df = pd.concat([overall_row, per_adr_rows], ignore_index=True)

    pd.set_option('display.max_rows', None)
    print('\n============================== Per-ADR Results ==============================')
    print(adr_df)
    pd.reset_option('display.max_rows')

    adr_output = (
        f'per_adr_metrics_'
        f'repeat{repeat}_seed{seed}.csv'
    )
    adr_df.to_csv(adr_output, index=False)
    print(f"Per-ADR metrics saved to '{adr_output}'")
    return rel_metrics, adr_df


# =========================================================
# Build / train / test
# =========================================================
model = models.MVN_DDI(
    n_atom_feats,
    n_atom_hid,
    kge_dim,
    rel_total,
    heads_out_feat_params=[64, 64, 64, 64],
    blocks_params=[2, 2, 2, 2],
)
loss = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda epoch: 0.96 ** epoch,
)
model.to(device=device)

train(
    model,
    train_data_loader,
    val_data_loader,
    loss,
    optimizer,
    n_epochs,
    device,
    scheduler,
)

test_model = torch.load(pkl_name)
test(test_data_loader, test_model)
