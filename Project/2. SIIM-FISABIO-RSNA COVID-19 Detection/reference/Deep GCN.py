# ==================================================== [ setting ] =====================================================
import numpy as np
import pandas as pd
import random

import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout
from torch_geometric.nn import ChebConv, NNConv, DeepGCNLayer
from torch_geometric.data import Data, DataLoader

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import copy

seed = 2109
train_file = "../data/train.json"
test_file = "../data/test.json"
bpps_top = "../data/bpps"

nb_flod = 5
device = "cuda"
batch_size = 16
epochs = 100
lr = 0.001

train_with_noisy_data = True
add_edge_for_paired_nodes = True
add_codon_nodes = True
T = 5

node_hidden_channels = 96
edge_hidden_channels = 16
hidden_channels3 = 32
num_layers = 10
dropout = 0.1

bpps_nb_mean = 0.077522
bpps_nb_std = 0.08914
error_mean_limit = 0.5

# ====================================================== [ def. function ] =============================================
def match_pair(structure):
    pair = [-1] * len(structure)
    pair_no = -1

    pair_no_stack = []
    for i, c in enumerate(structure):
        if c == '(':
            pair_no += 1
            pair[i] = pair_no
            pair_no_stack.append(pair_no)
        elif c == ')':
            pair[i] = pair_no_stack.pop()
    return pair

class MyData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, norm=None, face=None, weight=None, **kwargs):
        super(MyData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, norm=norm, face=face, **kwargs)
        self.weight = weight

def calc_error_mean(row):
    reactivity_error = row['reactivity_error']
    deg_error_Mg_pH10 = row['deg_error_Mg_pH10']
    deg_error_Mg_50C = row['deg_error_Mg_50C']

    return np.mean(np.abs(reactivity_error) + np.abs(deg_error_Mg_pH10) + np.abs(deg_error_Mg_50C)) / 3

def calc_sample_weight(row):
    if sample_is_clean(row):
        return 1.
    else:
        error_mean = calc_error_mean(row)
        if error_mean >= error_mean_limit :
            return 0.
        return 1. - error_mean / error_mean_limit

def add_edges(edge_index, edge_features, node1, node2, feature1, feature2):
    edge_index.append([node1, node2])
    edge_features.append(feature1)
    edge_index.append([node2, node1])
    edge_features.append(feature2)

def add_edges_between_base_nodes(edge_index, edge_features, node1, node2):
    edge_feature1 = [0, 0, 0,  1, 1, ]
    edge_feature2 = [0, 0, 0, -1,  1, ]
    add_edges(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)

def add_edges_between_paired_nodes(edge_index, edge_features, node1, node2, bpps_value):
    edge_feature1 = [1, 0, 0, 0, bpps_value, ]
    edge_feature2 = [1, 0, 0,  0, bpps_value, ]
    add_edges(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)

def add_edges_between_codon_nodes(edge_index, edge_features, node1, node2):
    edge_feature1 = [0, 0, 1, 1, 0, ]
    edge_feature2 = [0, 0, 1, -1, 0, ]
    add_edges(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)

def add_edges_between_codon_and_base_node(edge_index, edge_features, node1, node2):
    edge_feature1 = [0, 1, 0, 0, 0, ]
    edge_feature2 = [0, 1, 0, 0, 0, ]
    add_edges(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)

def add_node(node_features, feature):
    node_features.append(feature)

def add_base_node(node_features, sequence, predicted_loop_type, bpps_sum, bpps_nb):
    feature = [
        0,
        sequence == 'A',
        sequence == 'C',
        sequence == 'G',
        sequence == 'U',
        predicted_loop_type == 'S',
        predicted_loop_type == 'M',
        predicted_loop_type == 'I',
        predicted_loop_type == 'B',
        predicted_loop_type == 'H',
        predicted_loop_type == 'E',
        predicted_loop_type == 'X',
        bpps_sum,
        bpps_nb,
    ]
    add_node(node_features, feature)

def add_codon_node(node_features):
    feature = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
    add_node(node_features, feature)

def build_data(df, is_train):
    data = []
    for i in range(len(df)):
        targets = []
        node_features = []
        edge_features = []
        edge_index = []
        train_mask = []
        test_mask = []
        weights = []

        id = df.loc[i, 'id']
        path = os.path.join(bpps_top, id + '.npy')
        bpps = np.load(path)
        bpps_sum = bpps.sum(axis=0)
        sequence = df.loc[i, 'sequence']
        structure = df.loc[i, 'structure']
        pair_info = match_pair(structure)
        predicted_loop_type = df.loc[i, 'predicted_loop_type']
        seq_length = df.loc[i, 'seq_length']
        seq_scored = df.loc[i, 'seq_scored']
        bpps_nb = (bpps > 0).sum(axis=0) / seq_length
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        if is_train:
            sample_weight = calc_sample_weight(df.loc[i])
            reactivity = df.loc[i, 'reactivity']
            deg_Mg_pH10 = df.loc[i, 'deg_Mg_pH10']
            deg_Mg_50C = df.loc[i, 'deg_Mg_50C']

            for j in range(seq_length):
                if j < seq_scored:
                    targets.append([reactivity[j], deg_Mg_pH10[j], deg_Mg_50C[j], ])
                else:
                    targets.append([0, 0, 0])

        paired_nodes = {}
        for j in range(seq_length):
            add_base_node(node_features, sequence[j], predicted_loop_type[j], bpps_sum[j], bpps_nb[j])
            if j + 1 < seq_length:
                add_edges_between_base_nodes(edge_index, edge_features, j, j + 1)

            if pair_info[j] != -1:
                if pair_info[j] not in paired_nodes:
                    paired_nodes[pair_info[j]] = [j]
                else:
                    paired_nodes[pair_info[j]].append(j)

            train_mask.append(j < seq_scored)
            test_mask.append(True)
            if is_train:
                weights.append(sample_weight)

        if add_edge_for_paired_nodes:
            for pair in paired_nodes.values():
                bpps_value = bpps[pair[0], pair[1]]
                add_edges_between_paired_nodes(edge_index, edge_features, pair[0], pair[1], bpps_value)

        if add_codon_nodes:
            codon_node_idx = seq_length - 1
            for j in range(seq_length):
                if j % 3 == 0:
                    add_codon_node(node_features)
                    codon_node_idx += 1
                    train_mask.append(False)
                    test_mask.append(False)
                    if is_train:
                        weights.append(0)
                        targets.append([0, 0, 0])

                    if codon_node_idx > seq_length:
                        add_edges_between_codon_nodes(edge_index, edge_features, codon_node_idx - 1, codon_node_idx)
                add_edges_between_codon_and_base_node(edge_index, edge_features, j, codon_node_idx)

        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        if is_train:
            data.append(MyData(x=node_features, edge_index=edge_index, edge_attr=edge_features, train_mask=torch.tensor(train_mask), weight=torch.tensor(weights, dtype=torch.float), y=torch.tensor(targets, dtype=torch.float)))
        else:
            data.append(MyData(x=node_features, edge_index=edge_index, edge_attr=edge_features, test_mask=torch.tensor(test_mask)))
    return data

# ======================================================== [ modeling ] ================================================
class MapE2NxN(torch.nn.Module) :
    def __init__(self, in_channels, out_channels, hidden_channels) :
        super(MapE2NxN, self).__init__()
        self.linear1 = Linear(in_channels, hidden_channels)
        self.linear2 = Linear(hidden_channels, out_channels)
        self.dropout = Dropout(dropout)

    def forward(self, x) :
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MyDeeperGCN(torch.nn.Module) :
    def __init__(self, num_node_features, num_edge_features, node_hidden_channels, edge_hidden_channels, num_layers, num_classes) :
        super(MyDeeperGCN, self).__init__()
        self.node_encoder = ChebConv(num_node_features, node_hidden_channels, T)
        self.edge_enccoder = Linear(num_edge_features, edge_hidden_channels)
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1) :
            conv = NNConv(node_hidden_channels, node_hidden_channels, MapE2NxN(edge_hidden_channels, node_hidden_channels * node_hidden_channels, hidden_channels3))
            norm = LayerNorm(node_hidden_channels, elementwise_affine = True)
            act = ReLU(inplace = True)
            layer = DeepGCNLayer(conv, norm, act, block = "res+", dropout = dropout, ckpt_grad = i % 3)
            self.layers.append(layer)

        self.lin = Linear(node_hidden_channels, num_classes)
        self.dropout = Dropout(dropout)

    def forward(self, data) :
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        seq_edge_index = edge_index[:, edge_attr[:, 0] == 0]
        x = self.node_encoder(x, seq_edge_index)

        edge_attr = self.edge_enccoder(edge_attr)
        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1: ] :
            x = layer(x. edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = self.dropout(x)
        return self.lin(x)

def weighted_mse_loss(prds, tgts, weight) :
    return torch.mean(weight * (prds - tgts) ** 2)

def sample_is_clean(row) :
    return row["SN_filter"] == 1

def add_y_cat(df) :
    target_mean = df["reactivity"].apply(np.mean) + df["deg_Mg_pH10"].apply(np.mean) + df["deg_Mg_50C"].apply(np.mean)
    df["y_cat"] = pd.qcut(np.array(target_mean), q = 20).codes

# ======================================================== [ training ] ================================================
if __name__ == "__main__" :
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    df_tr = pd.read_json(train_file, lines = True)
    add_y_cat(df_tr)

    is_clean = df_tr.apply(sample_is_clean, axis = 1)
    df_clean = df_tr[is_clean].reset_index(drop = True)
    df_noisy = df_tr[is_clean == False].reset_index(drop = True)
    del df_tr

    all_ys = torch.zeros((0, 3)).to(device).detach()
    all_outs = torch.zeros((0, 3)).to(device).detach()
    best_model_states = []
    kf = StratifiedKFold(nb_flod, shuffle = True, random_state = seed)
    for fold, ((clean_train_idx, clean_valid_idx), (noisy_train_idx, noisy_valid_idx)) in enumerate(zip(kf.split(df_clean, df_clean["y_cat"]), kf.split(df_noisy, df_noisy["y_cat"]))) :
        df_train = df_clean.loc[clean_train_idx]
        if train_with_noisy_data :
            df_train_noisy = df_noisy.loc[noisy_train_idx]
            df_train_noisy = df_train_noisy[df_train_noisy.apply(calc_error_mean, axis = 1) <= error_mean_limit]
            df_train = df_train.append(df_train_noisy)

        data_train = build_data(df_train.reset_index(drop = True), True)
        del df_train
        loader_train = DataLoader(data_train, batch_size = batch_size, shuffle = True)

        df_valid_clean = df_clean.loc[clean_valid_idx].reset_index(drop = True)
        data_valid_clean = build_data(df_valid_clean, True)
        del df_valid_clean
        loader_valid_clean = DataLoader(data_valid_clean, batch_size = batch_size, shuffle = False)

        model = MyDeeperGCN(data_train[0].num_node_features, data_train[0].num_edge_features, node_hidden_channels = node_hidden_channels, edge_hidden_channels = edge_hidden_channels, num_layers = num_layers, num_classes = 3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        best_mcrmse = np.inf
        for epoch in range(epochs) :
            print("Epoch", epoch)
            model.train()
            train_loss = 0.0
            nb = 0
            for data in tqdm(loader_train) :
                data = data.to(device)
                mask = data.train_mask
                weight = data.weight[mask]
                optimizer.zero_grad()
                out = model(data)[mask]
                y = data.y[mask]
                loss = criterion(out, y, weight)
                loss.backward()
                train_loss += loss.item() * y.size(0)
                nb += y.size(0)
                del data, out, y, loss

            train_loss /= nb
            model.eval()
            valid_loss = 0.0
            nb = 0
            ys = torch.zeros((0, 3)).to(device).detach()
            outs = torch.zeros((0, 3)).to(device).detach()

            for data in tqdm(loader_valid_clean) :
                data = data.to(device)
                mask = data.train_mask

                out = model(data)[mask].detach()
                y = data.y[mask].detach()
                loss = criterion(out, y).detach()
                valid_loss ++ loss.item() * y.size(0)
                nb += y.size(0)

                outs = torch.cat((outs, out), dim = 0)
                ys = torch.cat((ys, y), dim = 0)
                del data, out, y, loss

            valid_loss /= nb
            mcrmse = criterion(outs, ys).item()

            print("T Loss: {:.4f} V Loss: {:.4f} V MCRMSE: {:.4f}".format(train_loss, valid_loss, mcrmse))
            if mcrmse < best_mcrmse:
                print('Best valid MCRMSE updated to', mcrmse)
                best_mcrmse = mcrmse
                best_model_state = copy.deepcopy(model.state_dict())

        del data_train
        del data_valid_clean
        best_model_states.append(best_model_state)

        model.load_state_dict(best_model_state)
        model.eval()
        for data in tqdm(loader_valid_clean):
            data = data.to(device)
            mask = data.train_mask

            out = model(data)[mask].detach()
            y = data.y[mask].detach()

            all_ys = torch.cat((all_ys, y), dim=0)
            all_outs = torch.cat((all_outs, out), dim=0)

            del data, out, y

    print('CV MCRMSE ', criterion(all_outs, all_ys).item())
    del all_outs, all_ys

    print('Predicting test data')
    print('Reading', test_file)
    df_te = pd.read_json(test_file, lines=True)
    data_test = build_data(df_te, False)
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    id_seqpos = build_id_seqpos(df_te)

    preds = torch.zeros((len(id_seqpos), 3)).to(device).detach()
    for best_model_state in best_model_states:
        model.load_state_dict(best_model_state)
        model.eval()
        outs = torch.zeros((0, 3)).to(device).detach()
        for data in tqdm(loader_test):
            data = data.to(device)
            mask = data.test_mask

            out = model(data)[mask].detach()
            outs = torch.cat((outs, out), dim=0)

            del data
            del out
        preds += outs

    preds /= len(best_model_states)
    preds = preds.cpu().numpy()

    df_sub = pd.DataFrame({'id_seqpos': id_seqpos, 'reactivity': preds[:,0], 'deg_Mg_pH10': preds[:,1], 'deg_pH10': 0, 'deg_Mg_50C': preds[:,2], 'deg_50C': 0})
    print('Writing submission.csv')
    df_sub.to_csv('submission.csv', index=False)

