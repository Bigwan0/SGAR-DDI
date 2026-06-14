import copy
from datetime import datetime
import time 
import argparse

import torch
import torch.nn.functional as F
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np

import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='inductive.pkl')

# Early stopping parameters
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
parser.add_argument('--min_delta', type=float, default=0.001, help='minimum improvement for early stopping')

# NEW: Separate model saving parameters
parser.add_argument('--s1_pkl_name', type=str, default='inductive_s1_best.pkl', help='best model for S1')
parser.add_argument('--s2_pkl_name', type=str, default='inductive_s2_best.pkl', help='best model for S2')

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
patience = args.patience
min_delta = args.min_delta
s1_pkl_name = args.s1_pkl_name
s2_pkl_name = args.s2_pkl_name
device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
############################################################

###### Dataset
df_ddi_train = pd.read_csv('inductive_data/fold1/train.csv')
df_ddi_s1 = pd.read_csv('inductive_data/fold1/s1.csv')
df_ddi_s2 = pd.read_csv('inductive_data/fold1/s2.csv')


train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1['d1'], df_ddi_s1['d2'], df_ddi_s1['type'])]
s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2['d1'], df_ddi_s2['d2'], df_ddi_s2['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
s1_data = DrugDataset(s1_tup, disjoint_split=True)
s2_data = DrugDataset(s2_tup, disjoint_split=True)

print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
s1_data_loader = DrugDataLoader(s1_data, batch_size=batch_size *3,num_workers=2)
s2_data_loader = DrugDataLoader(s2_data, batch_size=batch_size *3,num_workers=2)


CL_WEIGHT = 0.05
CL_MASK_RATIO = 0.10
CL_TEMPERATURE = 0.20


def _clone_data_obj(data):
    if hasattr(data, 'clone'):
        return data.clone()
    return copy.deepcopy(data)


def _mask_node_features(data, mask_ratio=CL_MASK_RATIO):
    data = _clone_data_obj(data)
    if (not hasattr(data, 'x')) or data.x is None or data.x.numel() == 0:
        return data

    x = data.x.clone()
    num_nodes = x.size(0)
    if num_nodes == 0:
        data.x = x
        return data

    num_mask = max(1, int(num_nodes * mask_ratio))
    perm = torch.randperm(num_nodes, device=x.device)[:num_mask]
    x[perm] = 0.0
    data.x = x
    return data


def _augment_positive_triple(pos_tri, mask_ratio=CL_MASK_RATIO):
    h_data, t_data, rels, b_graph = pos_tri
    h_aug = _mask_node_features(h_data, mask_ratio)
    t_aug = _mask_node_features(t_data, mask_ratio)
    b_aug = _clone_data_obj(b_graph)
    return h_aug, t_aug, rels, b_aug


def _info_nce(z1, z2, temperature=CL_TEMPERATURE):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.transpose(0, 1)) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_12 + loss_21)


def _mask_x_tensor(x, mask_ratio=CL_MASK_RATIO):
    x = x.detach().clone()
    if x.numel() == 0:
        return x

    num_nodes = x.size(0)
    if num_nodes == 0:
        return x

    num_mask = max(1, int(num_nodes * mask_ratio))
    perm = torch.randperm(num_nodes, device=x.device)[:num_mask]
    x[perm] = 0.0
    return x


def _compute_cl_loss(model, pos_tri, h_last_base, t_last_base, h_x_orig, t_x_orig,
                     mask_ratio=CL_MASK_RATIO, temperature=CL_TEMPERATURE):
    h_data, t_data, rels, b_graph = pos_tri

    h_proj = model.cl_proj(h_last_base.detach())
    t_proj = model.cl_proj(t_last_base.detach())

    h_saved = h_data.x
    t_saved = t_data.x

    try:
        h_data.x = _mask_x_tensor(h_x_orig, mask_ratio)
        t_data.x = _mask_x_tensor(t_x_orig, mask_ratio)

        _, h_last_aug, t_last_aug = model((h_data, t_data, rels, b_graph), return_last_repr=True)
    finally:
        h_data.x = h_saved
        t_data.x = t_saved

    h_proj_aug = model.cl_proj(h_last_aug)
    t_proj_aug = model.cl_proj(t_last_aug)

    loss_h = _info_nce(h_proj, h_proj_aug, temperature)
    loss_t = _info_nce(t_proj, t_proj_aug, temperature)
    return 0.5 * (loss_h + loss_t)


def do_compute(batch, device, model, compute_cl=False):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth, rel_types = [], [], []
    pos_tri, neg_tri = batch
    
    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    h_x_orig = pos_tri[0].x.detach().clone()
    t_x_orig = pos_tri[1].x.detach().clone()
    p_score, h_last_base, t_last_base = model(pos_tri, return_last_repr=True)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))
    # Ensure rel_types are properly squeezed to 1D array
    rel_types.append(pos_tri[2].squeeze().cpu().numpy())

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))
    # Ensure rel_types are properly squeezed to 1D array
    rel_types.append(neg_tri[2].squeeze().cpu().numpy())

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)
    # Flatten each array before concatenating if needed
    rel_types = [rt.flatten() if rt.ndim > 1 else rt for rt in rel_types]
    rel_types = np.concatenate(rel_types)

    if compute_cl:
        cl_loss = _compute_cl_loss(model, pos_tri, h_last_base, t_last_base, h_x_orig, t_x_orig)
        return p_score, n_score, probas_pred, ground_truth, rel_types, cl_loss

    return p_score, n_score, probas_pred, ground_truth, rel_types

def do_compute_metrics(probas_pred, target, rel_types=None, per_rel=False):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

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


def train(model, train_data_loader, s1_data_loader, s2_data_loader, loss_fn, optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())
    
    # Separate tracking for S1 and S2 best models
    best_s1_acc = 0
    best_s2_acc = 0
    best_s1_epoch = 0
    best_s2_epoch = 0
    patience_counter = 0
    
    for i in range(1, n_epochs+1):
        start = time.time()
        train_loss = 0 
        s1_loss = 0
        s2_loss = 0
      
        train_probas_pred = []
        train_ground_truth = []
        train_rel_types = []

        s1_probas_pred = []
        s1_ground_truth = []
        s1_rel_types = []

        s2_probas_pred = []
        s2_ground_truth = []
        s2_rel_types = []

        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth, rel_types, cl_loss = do_compute(batch, device, model, compute_cl=True)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            train_rel_types.append(rel_types)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)
            
            optimizer.zero_grad()
            ddi_loss = loss
            loss = ddi_loss + CL_WEIGHT * cl_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)
            train_rel_types = np.concatenate(train_rel_types)

            train_acc, train_auc_roc, train_f1, train_precision, train_recall, train_int_ap, train_ap = do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in s1_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth, rel_types = do_compute(batch, device, model)
                s1_probas_pred.append(probas_pred)
                s1_ground_truth.append(ground_truth)
                s1_rel_types.append(rel_types)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                s1_loss += loss.item() * len(p_score)            

            s1_loss /= len(s1_data)
            s1_probas_pred = np.concatenate(s1_probas_pred)
            s1_ground_truth = np.concatenate(s1_ground_truth)
            s1_rel_types = np.concatenate(s1_rel_types)
            s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = do_compute_metrics(s1_probas_pred, s1_ground_truth)
        
            for batch in s2_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth, rel_types = do_compute(batch, device, model)
                s2_probas_pred.append(probas_pred)
                s2_ground_truth.append(ground_truth)
                s2_rel_types.append(rel_types)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                s2_loss += loss.item() * len(p_score)            

            s2_loss /= len(s2_data)
            s2_probas_pred = np.concatenate(s2_probas_pred)
            s2_ground_truth = np.concatenate(s2_ground_truth)
            s2_rel_types = np.concatenate(s2_rel_types)
            s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = do_compute_metrics(s2_probas_pred, s2_ground_truth)

            # UPDATED: Track improvements for both S1 and S2 separately
            s1_improved = False
            s2_improved = False
            overall_improved = False
            
            # Check S1 improvement
            if s1_acc > best_s1_acc + min_delta:
                best_s1_acc = s1_acc
                best_s1_epoch = i
                s1_improved = True
                overall_improved = True
                torch.save(model, s1_pkl_name)
                print(f"*** New best S1 model saved at epoch {i} with S1 acc: {s1_acc:.4f} ***")
            
            # Check S2 improvement  
            if s2_acc > best_s2_acc + min_delta:
                best_s2_acc = s2_acc
                best_s2_epoch = i
                s2_improved = True
                overall_improved = True
                torch.save(model, s2_pkl_name)
                print(f"*** New best S2 model saved at epoch {i} with S2 acc: {s2_acc:.4f} ***")
            
            # Update patience counter based on overall improvement
            if overall_improved:
                patience_counter = 0
            else:
                patience_counter += 1
               
        if scheduler:
            scheduler.step()

        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, s1_loss: {s1_loss:.4f}, s2_loss: {s2_loss:.4f}')
        print(f'\t\ttrain_acc: {train_acc:.4f}, train_roc: {train_auc_roc:.4f}, train_precision: {train_precision:.4f}, train_recall: {train_recall:.4f}')
        print(f'\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_precision: {s1_precision:.4f}, s1_recall: {s1_recall:.4f}')
        print(f'\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_precision: {s2_precision:.4f}, s2_recall: {s2_recall:.4f}')
        if hasattr(model, 'blocks'):
            gamma_vals = []
            for bi, blk in enumerate(model.blocks):
                if hasattr(blk, 'raw_gamma'):
                    gv = torch.nn.functional.softplus(blk.raw_gamma).item()
                    gamma_vals.append(f'b{bi}:{gv:.4f}')
            if gamma_vals:
                print('\t\tgamma(s): ' + ', '.join(gamma_vals))
        
        if overall_improved:
            improvement_msg = []
            if s1_improved:
                improvement_msg.append("S1")
            if s2_improved:
                improvement_msg.append("S2")
            print(f'\t\t*** IMPROVEMENT ({"/".join(improvement_msg)}) *** (Patience: {patience_counter}/{patience})')
        else:
            print(f'\t\tNo improvement (Patience: {patience_counter}/{patience})')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {i}!')
            print(f'Best S1 model was at epoch {best_s1_epoch} with S1 acc: {best_s1_acc:.4f}')
            print(f'Best S2 model was at epoch {best_s2_epoch} with S2 acc: {best_s2_acc:.4f}')
            break

    print(f'\nTraining completed.')
    print(f'Best S1 model: epoch {best_s1_epoch}, S1 acc: {best_s1_acc:.4f}')
    print(f'Best S2 model: epoch {best_s2_epoch}, S2 acc: {best_s2_acc:.4f}')


def test_single_dataset(data_loader, model, dataset_name):
    probas_pred = []
    ground_truth = []
    rel_types = []
    
    with torch.no_grad():
        for batch in data_loader:
            model.eval()
            p_score, n_score, batch_probas_pred, batch_ground_truth, batch_rel_types = do_compute(batch, device, model=model)
            probas_pred.append(batch_probas_pred)
            ground_truth.append(batch_ground_truth)
            rel_types.append(batch_rel_types)
      
        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)
        rel_types = np.concatenate(rel_types)
        
        # Get overall metrics
        acc, auc_roc, f1, precision, recall, int_ap, ap = do_compute_metrics(probas_pred, ground_truth)
        
        # Get per-relation metrics
        _, _, _, _, _, _, _, rel_metrics = do_compute_metrics(probas_pred, ground_truth, rel_types, per_rel=True)

    print(f'{dataset_name} Results (using best {dataset_name} model):')
    print(f'\t\t{dataset_name.lower()}_acc: {acc:.4f}, {dataset_name.lower()}_roc: {auc_roc:.4f}, {dataset_name.lower()}_f1: {f1:.4f}, {dataset_name.lower()}_precision: {precision:.4f}, {dataset_name.lower()}_recall: {recall:.4f}, {dataset_name.lower()}_int_ap: {int_ap:.4f}, {dataset_name.lower()}_ap: {ap:.4f}')
    
    return {
        'acc': acc, 'auroc': auc_roc, 'f1': f1, 'precision': precision, 
        'recall': recall, 'int_ap': int_ap, 'ap': ap, 'rel_metrics': rel_metrics,
        'dataset': dataset_name, 'probas_pred': probas_pred, 'ground_truth': ground_truth, 'rel_types': rel_types
    }


def test_separate_models(s1_data_loader, s2_data_loader):
    print('\n')
    print('============================== Testing with Separate Best Models ==============================')
    
    # Test S1 with best S1 model
    print('Testing S1 with best S1 model...')
    s1_best_model = torch.load(s1_pkl_name)
    s1_results = test_single_dataset(s1_data_loader, s1_best_model, 'S1')
    
    # Test S2 with best S2 model  
    print('Testing S2 with best S2 model...')
    s2_best_model = torch.load(s2_pkl_name)
    s2_results = test_single_dataset(s2_data_loader, s2_best_model, 'S2')
    
    # Create combined per-ADR results table
    print('\n')
    print('============================== Per-ADR Results ==============================')
    
    # Create a combined DataFrame for both S1 and S2 results
    adr_data = []
    
    # Add S1 overall metrics first
    s1_res = s1_results
    adr_data.append({
        'ADR_Type': 'Overall_S1',
        'Accuracy': s1_res['acc'],
        'AUROC': s1_res['auroc'],
        'F1': s1_res['f1'],
        'Precision': s1_res['precision'],
        'Recall': s1_res['recall'],
        'Int_AP': s1_res['int_ap'],
        'AP': s1_res['ap'],
        'Dataset': 'S1'
    })
    
    # Add S1 per-ADR metrics
    for rel, metrics_dict in s1_res['rel_metrics'].items():
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
                'Dataset': 'S1'
            })
        else:
            error_msg = metrics_dict.get('error', 'Unknown error') if isinstance(metrics_dict, dict) else 'Unknown error'
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
                'Dataset': 'S1'
            })
    
    # Add S2 overall metrics
    s2_res = s2_results
    adr_data.append({
        'ADR_Type': 'Overall_S2',
        'Accuracy': s2_res['acc'],
        'AUROC': s2_res['auroc'],
        'F1': s2_res['f1'],
        'Precision': s2_res['precision'],
        'Recall': s2_res['recall'],
        'Int_AP': s2_res['int_ap'],
        'AP': s2_res['ap'],
        'Dataset': 'S2'
    })
    
    # Add S2 per-ADR metrics
    for rel, metrics_dict in s2_res['rel_metrics'].items():
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
                'Dataset': 'S2'
            })
        else:
            error_msg = metrics_dict.get('error', 'Unknown error') if isinstance(metrics_dict, dict) else 'Unknown error'
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
                'Dataset': 'S2'
            })
    
    # Create DataFrame with the desired ordering:
    # Overall S1, then per-ADR S1 (sorted by accuracy desc, then ADR_Type asc)
    # Overall S2, then per-ADR S2 (sorted by accuracy desc, then ADR_Type asc)
    adr_df = pd.DataFrame(adr_data)
    
    # Split into S1 and S2 data
    s1_data = adr_df[adr_df['Dataset'] == 'S1'].copy()
    s2_data = adr_df[adr_df['Dataset'] == 'S2'].copy()
    
    # For S1: Overall first, then per-ADR sorted by accuracy and ADR_Type
    s1_overall = s1_data[s1_data['ADR_Type'] == 'Overall_S1']
    s1_per_adr = s1_data[s1_data['ADR_Type'] != 'Overall_S1'].sort_values(by=['Accuracy', 'ADR_Type'], ascending=[False, True])
    s1_ordered = pd.concat([s1_overall, s1_per_adr], ignore_index=True)
    
    # For S2: Overall first, then per-ADR sorted by accuracy and ADR_Type
    s2_overall = s2_data[s2_data['ADR_Type'] == 'Overall_S2']
    s2_per_adr = s2_data[s2_data['ADR_Type'] != 'Overall_S2'].sort_values(by=['Accuracy', 'ADR_Type'], ascending=[False, True])
    s2_ordered = pd.concat([s2_overall, s2_per_adr], ignore_index=True)
    
    # Combine: S1 results first, then S2 results
    final_adr_df = pd.concat([s1_ordered, s2_ordered], ignore_index=True)
    
    # Display the DataFrame
    pd.set_option('display.max_rows', None)  # Show all rows
    print(final_adr_df)
    
    # Reset display options
    pd.reset_option('display.max_rows')
    
    # Save to CSV
    final_adr_df.to_csv('per_adr_metrics_separate_models.csv', index=False)
    print("Per-ADR metrics saved to 'per_adr_metrics_separate_models.csv'")
    
    return s1_results, s2_results, final_adr_df


model = models.MVN_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2])
loss = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
# print(model)
model.to(device=device)

# if __name__ == '__main__':
# Train with separate model saving
train(model, train_data_loader, s1_data_loader, s2_data_loader, loss, optimizer, n_epochs, device, scheduler)

# Test with separate best models
s1_results, s2_results, adr_df = test_separate_models(s1_data_loader, s2_data_loader)