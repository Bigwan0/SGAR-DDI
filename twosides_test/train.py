from datetime import datetime
import time 
import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast, GradScaler
from torch import optim
import torch.nn.functional as F
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader, TOTAL_ATOM_FEATS
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=963, help='num of interaction types')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')


parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='transductive_twosides.pkl')
parser.add_argument('--fold', type=int, default=1, choices=[0,1,2], help='which twosides fold to run')

args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
pkl_name = args.pkl_name
fold = args.fold

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
############################################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TWOSIDES_DIR = PROJECT_ROOT / "twosides"

# -------- speedup config --------
USE_AMP = True
VAL_INTERVAL = 1
PATIENCE = 10
SKIP_PER_REL = True

# -------- CL config --------
CL_WEIGHT = 0.05
CL_MASK_RATIO = 0.10
CL_TEMPERATURE = 0.20

###### Dataset
def split_train_valid(data, fold, val_ratio=0.2):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
    train_tup = data[train_index]
    val_tup = data[val_index]
    train_tup = [(tup[0],tup[1],int(tup[2]),tup[3])for tup in train_tup ]
    val_tup = [(tup[0],tup[1],int(tup[2]),tup[3])for tup in val_tup ]

    return train_tup, val_tup

df_ddi_train = pd.read_csv(TWOSIDES_DIR / f'fold{fold}' / 'train.csv')
df_ddi_test = pd.read_csv(TWOSIDES_DIR / f'fold{fold}' / 'test.csv')




train_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'], df_ddi_train['Neg samples'])]
train_tup, val_tup = split_train_valid(train_tup, fold, val_ratio=0.2)
test_tup = [(h, t, r, n) for h, t, r, n in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'], df_ddi_test['Neg samples'])]

train_data = DrugDataset(train_tup)
val_data = DrugDataset(val_tup)
test_data = DrugDataset(test_tup)


print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=8)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size *3,num_workers=8)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3,num_workers=8)



def _info_nce(z1, z2, temperature=CL_TEMPERATURE):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


def _mask_x_tensor(x, mask_ratio=CL_MASK_RATIO):
    if mask_ratio <= 0:
        return x
    out = x.clone()
    node_mask = (torch.rand(out.size(0), device=out.device) < mask_ratio).unsqueeze(-1)
    out = out.masked_fill(node_mask, 0.0)
    return out




def _compute_cl_loss(model, pos_tri_raw, h_last_base, t_last_base,
                     mask_ratio=CL_MASK_RATIO, temperature=CL_TEMPERATURE):
    h_data_raw, t_data_raw, rels, b_graph_raw = pos_tri_raw

    h_data_aug = h_data_raw.clone()
    t_data_aug = t_data_raw.clone()
    b_graph_aug = b_graph_raw.clone()

    h_data_aug.x = _mask_x_tensor(h_data_aug.x, mask_ratio)
    t_data_aug.x = _mask_x_tensor(t_data_aug.x, mask_ratio)

    _, h_last_aug, t_last_aug = model((h_data_aug, t_data_aug, rels, b_graph_aug), return_last_repr=True)

    h_proj = model.cl_proj(h_last_base)
    t_proj = model.cl_proj(t_last_base)
    h_proj_aug = model.cl_proj(h_last_aug)
    t_proj_aug = model.cl_proj(t_last_aug)

    loss_h = _info_nce(h_proj, h_proj_aug, temperature)
    loss_t = _info_nce(t_proj, t_proj_aug, temperature)
    return 0.5 * (loss_h + loss_t)



def do_compute(batch, device, model, compute_cl=False):
    """
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    """
    probas_pred, ground_truth, rel_types = [], [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]

    # keep raw graph copies BEFORE the main forward mutates graph features
    h_data_raw = pos_tri[0].clone()
    t_data_raw = pos_tri[1].clone()
    rels_raw = pos_tri[2]
    b_graph_raw = pos_tri[3].clone()

    if compute_cl:
        p_score, h_last_base, t_last_base = model(pos_tri, return_last_repr=True)
        cl_loss = _compute_cl_loss(model, (h_data_raw, t_data_raw, rels_raw, b_graph_raw), h_last_base, t_last_base)
    else:
        p_score = model(pos_tri)
        cl_loss = torch.tensor(0.0, device=device)

    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))
    rel_types.append(pos_tri[2].squeeze().cpu().numpy())

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))
    rel_types.append(neg_tri[2].squeeze().cpu().numpy())

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)
    rel_types = np.concatenate(rel_types)

    return p_score, n_score, probas_pred, ground_truth, rel_types, cl_loss


def do_compute_metrics(probas_pred, target, rel_types=None, per_rel=False):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)

    # If we want per-relation metrics and relation types are provided
    if per_rel and rel_types is not None:
        # Create a dictionary to store metrics per relation type
        rel_metrics = {}
        unique_rels = np.unique(rel_types)
        
        for rel in unique_rels:
            rel_mask = rel_types == rel
            # Convert target to numpy array if it's not already
            target_array = np.array(target)
            # Check if we have positive and negative examples
            if sum(rel_mask) > 0 and sum(target_array[rel_mask]) > 0 and sum(1 - target_array[rel_mask]) > 0:
                try:
                    rel_pred = pred[rel_mask]
                    rel_target = target_array[rel_mask]
                    rel_probas = probas_pred[rel_mask]
                    
                    rel_acc = metrics.accuracy_score(rel_target, rel_pred)
                    rel_auroc = metrics.roc_auc_score(rel_target, rel_probas)
                    rel_f1 = metrics.f1_score(rel_target, rel_pred)
                    rel_precision = metrics.precision_score(rel_target, rel_pred)
                    rel_recall = metrics.recall_score(rel_target, rel_pred)
                    rel_p, rel_r, rel_t = metrics.precision_recall_curve(rel_target, rel_probas)
                    rel_int_ap = metrics.auc(rel_r, rel_p)
                    rel_ap = metrics.average_precision_score(rel_target, rel_probas)
                    
                    rel_metrics[rel] = {
                        'acc': rel_acc,
                        'auroc': rel_auroc,
                        'f1': rel_f1,
                        'precision': rel_precision,
                        'recall': rel_recall,
                        'int_ap': rel_int_ap,
                        'ap': rel_ap
                    }
                except Exception as e:
                    # Handle cases where metrics cannot be calculated
                    rel_metrics[rel] = {'error': str(e)}
            else:
                rel_metrics[rel] = {'error': 'Insufficient samples for both classes'}
        
        return acc, auroc, f1_score, precision, recall, int_ap, ap, rel_metrics
    
    return acc, auroc, f1_score, precision, recall, int_ap, ap



def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, device, scheduler=None):
    best_val_acc = -1.0
    patience_counter = 0
    scaler = GradScaler(enabled=(USE_AMP and str(device).startswith('cuda')))
    print('Starting training at', datetime.today())

    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        train_probas_pred = []
        train_ground_truth = []
        train_rel_types = []

        for batch in train_data_loader:
            model.train()
            optimizer.zero_grad()

            with autocast(enabled=(USE_AMP and str(device).startswith('cuda'))):
                p_score, n_score, probas_pred, ground_truth, rel_types, cl_loss = do_compute(batch, device, model, compute_cl=True)
                ddi_loss, loss_p, loss_n = loss_fn(p_score, n_score)
                loss = ddi_loss + CL_WEIGHT * cl_loss

            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            train_rel_types.append(rel_types)

            if USE_AMP and str(device).startswith('cuda'):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * len(p_score)

        train_loss /= len(train_data)
        train_probas_pred = np.concatenate(train_probas_pred)
        train_ground_truth = np.concatenate(train_ground_truth)
        train_rel_types = np.concatenate(train_rel_types)

        (train_acc, train_auc_roc, train_f1, train_precision,
         train_recall, train_int_ap, train_ap) = do_compute_metrics(train_probas_pred, train_ground_truth)

        run_val = (i % VAL_INTERVAL == 0) or (i == n_epochs)

        if run_val:
            val_loss = 0
            val_probas_pred = []
            val_ground_truth = []
            val_rel_types = []

            with torch.no_grad():
                for batch in val_data_loader:
                    model.eval()
                    with autocast(enabled=(USE_AMP and str(device).startswith('cuda'))):
                        p_score, n_score, probas_pred, ground_truth, rel_types, _ = do_compute(batch, device, model, compute_cl=False)
                        loss, loss_p, loss_n = loss_fn(p_score, n_score)

                    val_probas_pred.append(probas_pred)
                    val_ground_truth.append(ground_truth)
                    val_rel_types.append(rel_types)
                    val_loss += loss.item() * len(p_score)

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_rel_types = np.concatenate(val_rel_types)

            (val_acc, val_auc_roc, val_f1, val_precision,
             val_recall, val_int_ap, val_ap) = do_compute_metrics(val_probas_pred, val_ground_truth)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model, pkl_name)
                improved = True
            else:
                patience_counter += 1
                improved = False

            print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
            print(f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}')
            if improved:
                print(f'\t\t*** VAL IMPROVED: saved to {pkl_name} ***')
            else:
                print(f'\t\tNo improvement. Patience: {patience_counter}/{PATIENCE}')

            if patience_counter >= PATIENCE:
                print(f'Early stopping triggered at epoch {i}.')
                break
        else:
            print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} [validation skipped]')
            print(f'\t\ttrain_roc: {train_auc_roc:.4f}, train_precision: {train_precision:.4f}')

        if scheduler:
            scheduler.step()


def test(test_data_loader, model):
    test_probas_pred = []
    test_ground_truth = []
    test_rel_types = []

    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            with autocast(enabled=(USE_AMP and str(device).startswith('cuda'))):
                p_score, n_score, probas_pred, ground_truth, rel_types, _ = do_compute(batch, device, model, compute_cl=False)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
            test_rel_types.append(rel_types)

        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_rel_types = np.concatenate(test_rel_types)

        (test_acc, test_auc_roc, test_f1, test_precision,
         test_recall, test_int_ap, test_ap) = do_compute_metrics(test_probas_pred, test_ground_truth)

    print('\n')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')
    print('\n[skip_per_rel] Per-ADR / per-relation metrics skipped for speed.')

model = models.MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
loss = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
# print(model)
model.to(device=device)
# # if __name__ == '__main__':
train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device, scheduler)
test_model = torch.load(pkl_name)
test(test_data_loader,test_model)