from datetime import datetime
import time 
import argparse
import torch
import torch.nn.functional as F
import warnings

from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader, load_fg_enrichment_scores, precompute_functional_groups

warnings.filterwarnings('ignore',category=UserWarning)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')


parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='transductive_drugbank.pkl')

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
device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)

# Load functional group enrichment scores
print("Loading functional group enrichment scores...")
load_fg_enrichment_scores()
print("Precomputing functional groups for all molecules...")
precompute_functional_groups()
print("Functional group setup complete.")

############################################################
###### Dataset
def split_train_valid(data, fold, val_ratio=0.2):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
    train_tup = data[train_index]
    val_tup = data[val_index]
    train_tup = [(tup[0],tup[1],int(tup[2]))for tup in train_tup ]
    val_tup = [(tup[0],tup[1],int(tup[2]))for tup in val_tup ]

    return train_tup, val_tup

df_ddi_train = pd.read_csv('drugbank/fold1/train.csv')
df_ddi_test = pd.read_csv('drugbank/fold1/test.csv')

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
train_tup, val_tup = split_train_valid(train_tup,2, val_ratio=0.2)
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)


print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size *3,num_workers=2)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3,num_workers=2)



CL_WEIGHT = 0.05
CL_MASK_RATIO = 0.10
CL_TEMPERATURE = 0.20


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
    rel_types.append(pos_tri[2].squeeze().cpu().numpy())

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))
    rel_types.append(neg_tri[2].squeeze().cpu().numpy())

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)
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
    max_acc = 0
    print('Starting training at', datetime.today())
    for i in range(1, n_epochs+1):
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        train_rel_types = []
        val_probas_pred = []
        val_ground_truth = []
        val_rel_types = []
       
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

            for batch in val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth, rel_types, cl_loss = do_compute(batch, device, model, compute_cl=True)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                val_rel_types.append(rel_types)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)            

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_rel_types = np.concatenate(val_rel_types)
            val_acc, val_auc_roc, val_f1, val_precision, val_recall, val_int_ap, val_ap = do_compute_metrics(val_probas_pred, val_ground_truth)
            
            if val_acc > max_acc:
                max_acc = val_acc
                torch.save(model, pkl_name)
               
        if scheduler:
            scheduler.step()

        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
        f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
        print(f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}')

def test(test_data_loader, model):
    test_probas_pred = []
    test_ground_truth = []
    test_rel_types = []
    
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth, rel_types, cl_loss = do_compute(batch, device, model, compute_cl=True)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
            test_rel_types.append(rel_types)
            
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_rel_types = np.concatenate(test_rel_types)
        
        # Get overall metrics
        test_metrics = do_compute_metrics(test_probas_pred, test_ground_truth)
        test_acc, test_auc_roc, test_f1, test_precision, test_recall, test_int_ap, test_ap = test_metrics
        
        # Get per-relation metrics
        _, _, _, _, _, _, _, rel_metrics = do_compute_metrics(
            test_probas_pred, test_ground_truth, test_rel_types, per_rel=True)
            
    print('\n')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')
    
    print('\n')
    print('============================== Per-ADR Results ==============================')
    
    # Create a DataFrame for the per-ADR metrics
    adr_data = []
    
    # Add overall metrics first
    adr_data.append({
        'ADR_Type': 'Overall',
        'Accuracy': test_acc,
        'AUROC': test_auc_roc,
        'F1': test_f1,
        'Precision': test_precision,
        'Recall': test_recall,
        'Int_AP': test_int_ap,
        'AP': test_ap
    })
    
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
                'AP': metrics_dict['ap']
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
                'Error': error_msg
            })
    
    # Create DataFrame and sort: Overall first, then per-ADR by Accuracy (descending) and ADR_Type
    adr_df = pd.DataFrame(adr_data)
    overall_row = adr_df[adr_df['ADR_Type'] == 'Overall']
    per_adr_rows = adr_df[adr_df['ADR_Type'] != 'Overall'].sort_values(by=['Accuracy', 'ADR_Type'], ascending=[False, True])
    adr_df = pd.concat([overall_row, per_adr_rows], ignore_index=True)
    
    # Display the DataFrame
    pd.set_option('display.max_rows', None)  # Show all rows
    print(adr_df)
    
    # Reset display options
    pd.reset_option('display.max_rows')
    
    # Save to CSV
    adr_df.to_csv('per_adr_metrics.csv', index=False)
    print("Per-ADR metrics saved to 'per_adr_metrics.csv'")
    
    return rel_metrics, adr_df


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