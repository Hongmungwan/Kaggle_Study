# ===================================================== [ setting ] ====================================================
import os
from glob import glob
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, global_add_pool
from torch_geometric.data import DataLoader, InMemoryDataset

import rdkit
import rdkit.Chem as Chem
from collections import defaultdict

train = pd.read_csv("./train.csv")
dev = pd.read_csv("./dev.csv")
test = pd.read_csv("./test.csv")

full = pd.concat([train, dev, test], axis = 0, ignore_index = True)
full["y"] = full["S1_energy(eV)"] - full["T1_energy(eV"]
full["folder"] = full["uid"].apply(lambda x : x.split("_")[0])

# ================================================== [ preprocessing ] =================================================
## + todo [ Convert Diction type ] =================
def create_encoders(df) :
    encoder_atom = defaultdict(lambda : len(encoder_atom))
    encoder_bond_type = defaultdict(lambda : len(encoder_bond_type))
    encoder_bond_stereo = defaultdict(lambda : len(encoder_bond_stereo))
    encoder_bond_type_stereo = defaultdict(lambda : len(encoder_bond_type_stereo))

    target = df["SMILES"].values
    total_num = len(target)
    for i, smiles in enumerate(target) :
        print(f'Creating the label encoders for atoms, bond_type, and bond_stereo ... [{i + 1} / {total_num}] done !', end = "\r")
        m = Chem.MolFromSmiles(smiles)
        m = Chem.AddHs(m)
        for atom in m.GetAtoms() :
            encoder_atom[atom.GetAtomicNum()]

        for bond in m.GetBonds() :
            encoder_bond_type[bond.GetBondTpeAsDouble()]
            encoder_bond_stereo[bond.GetStereo()]
            encoder_bond_type_stereo[(bond.GetBondTpeAsDouble(), bond.GetStereo())]
    return encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo

encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo = create_encoders(full)

## + todo [ Convert torch groph model input ] =======
def row2data(row, encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo) :
    smiles = row.SMILES
    y = row.y
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    ## + todo [ Creating node feature vector ]
    num_nodes = len(list(m.GetAtoms()))
    x = np.zeros((num_nodes, len(encoder_atom.keys())))
    for i in m.GetAtoms() :
        x[i.GetIdx(), encoder_atom[i.GetAtomicNum()]] = 1
    x = torch.from_numpy(x).float()

    ## + todo [ Creating edge_index and edge_type ]
    i = 0
    num_edges = 2 * len(list(m.GetBonds()))
    edge_index = np.zeros((2, num_edges), dtype = np.int64)
    edge_type = np.zeros((num_edges), dtype = np.int64)
    for edge in m.GetBonds() :
        u = min(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        v = max(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        bond_type = edge.GetBondTypeAsDouble()
        bond_stereo = edge.GetStereo()
        bond_label = encoder_bond_type_stereo([bond_type, bond_stereo])

        edge_index[0, i] = u
        edge_index[1, i] = v
        edge_index[0, i + 1] = v
        edge_index[1, i + 1] = u
        edge_type[i] = bond_label
        edge_type[i + 1] = bond_label
        i += 2

    edge_index = torch.from_numpy(edge_index)
    edge_type = torch.from_numpy((edge_type))
    y = torch.tensor([y]).float()

    data = Data(x = x, dege_index = edge_index, edge_type = edge_type, y = y, uid = row.uid)
    return data

for i, row in full.iterrows() :
    print(f'Converting each data into torch.Data ... [{i + 1} / {len(full)}] done !', end='\r')
    data = row2data(row, encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo)
    fpath = f'./output/rgcn/{row.folder}/{row.uid}.pt'
    torch.save(data, fpath)

# ===================================================== [ modeling ] ===================================================
seed = 2109
num_node_features = 13
node_embedding_dim = 256
hidden_channels = (256, 256, 256, 256, 256, 256, 256, 256)
hidden_dims = (1024, 512)
dropout = 0.3

lr = 0.0001
n_epochs = 300
batch_size = 128

class GNNDataset(InMemoryDataset) :
    def __init__(self, root) :
        super(GNNDataset, self).__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) :
        return []

    @property
    def processed_files(self) :
        return ["data.pt"]

    def download(self) :
        pass

    def process(self) :
        data_list = glob(f'{self.root}/*.pt')
        data_list = list(map(torch.load, data_list))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class NodeEncoder(nn.Module) :
    def __init__(self, input_dim, embedding_dim) :
        super(NodeEncoder, self).__init__()
        self.encoder = nn.Linear(input_dim, embedding_dim)

    def forward(self, x) :
        return self.encoder(x)

class RGCNSkipConnection(torch.nn.Module) :
    def __init__(self, hidden_channels, hidden_dims, num_node_features = 13, node_embedding_dim = 256, dropout = 0.3) :
        super(RGCNSkipConnection, self).__init__()
        self.node_encoder = NodeEncoder(num_node_features, node_embedding_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv_layers = nn.ModuleList()
        for i in range(len(hidden_channels)) :
            if i == 0 :
                conv = RGCNConv(node_embedding_dim, hidden_channels[i], num_relations = 6, aggr = "add")
            else :
                conv = RGCNConv(hidden_channels[i -1], hidden_channels[i], num_relations = 6, aggr = "add")
            self.conv_layers.append(conv)
        self.graph_pooling = nn.Sequential(nn.Linear(hidden_channels[-1], hidden_channels[-1]), nn.ReLU(), nn.Linear(hidden_channels[-1], hidden_channels[-1]),)

        self.fc_layers = nn.ModuleList()
        for i in range(len(hidden_dims)) :
            if i == 0 :
                fc = nn.Linear(hidden_channels[-1], hidden_dims[i])
            else :
                fc = nn.Linear(hidden_dims[i - 1], hidden_dims[i])
            self.fc_layers.append(fc)
        self.out = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x, edge_index, edge_type, batch) :
        x = self.node_encoder(x)
        for i, conv in enumerate(self.conv_layers) :
            skip = x
            x = conv(x, edge_index, edge_type)
            x = self.prelu(x + skip)
            x = F.normalize(x, 2, 1)

        x = self.graph_pooling(x)
        x = global_add_pool(x, batch)

        for i, fc in enumerate(self.fc_layers) :
            x = fc(x)
            x = self.dropout(x)
            x = F.relu(x)

        x = F.relu(self.out(x))
        return x

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class RGCNSolver:
    def __init__(self, model, lr, n_epochs, device=None):
        self.model = model
        self.device = device
        self.n_epochs = n_epochs

        self.criterion = torch.nn.L1Loss()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=lr)
        self.model.to(self.device)
        self.history = {'train_loss': [], 'valid_loss': [], 'dev_loss': []}

    def fit(self, train_loader, valid_loader, dev_loader):
        for epoch in range(self.n_epochs):
            t = time.time()
            train_loss = self.train_one_epoch(train_loader)

            valid_loss = self.evaluate(valid_loader)
            dev_loss = self.evaluate(dev_loader)

            message = f"[Epoch {epoch}] "
            message += f"Elapsed time: {time.time() - t:.3f} | "
            message += f"Train loss: {train_loss.avg:.5f} | "
            message += f"Validation loss: {valid_loss.avg:.5f} | "
            message += f"Dev loss: {dev_loss.avg:.5f} |"
            print(message)

            self.history['train_loss'].append(train_loss.avg)
            self.history['valid_loss'].append(valid_loss.avg)
            self.history['dev_loss'].append(dev_loss.avg)

    def train_one_epoch(self, loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()

        for step, data in enumerate(loader):
            print(f'Train step {(step + 1)} / {len(loader)} | ' + f'Summary loss: {summary_loss.avg:.5f} |' + f'Time: {time.time() - t:.3f} |', end='\r')
            data.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(data.x, data.edge_index, data.edge_type, data.batch)
            loss = self.criterion(y_pred, data.y.unsqueeze(1))
            loss.backward()
            self.optimizer.step()

            summary_loss.update(loss.detach().item(), data.num_graphs)
        return summary_loss

    def evaluate(self, loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()

        with torch.no_grad():
            for step, data in enumerate(loader):
                data.to(self.device)
                y_pred = self.model(data.x, data.edge_index, data.edge_type, data.batch)
                loss = self.criterion(y_pred, data.y.unsqueeze(1))
                summary_loss.update(loss.detach().item(), data.num_graphs)
        return summary_loss

torch.manual_seed(2109)
dataset = GNNDataset(f'../outputs/rgcn/train')
train_dataset = dataset[:27000]
valid_dataset = dataset[27000:]
dev_dataset = GNNDataset("../outputs/rgcn/dev")
test_dataset = GNNDataset("../outputs/rgcn/test")

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
dev_loader = DataLoader(dev_dataset, batch_size = 1, shuffle = False)

model = RGCNSkipConnection(hidden_channels, hidden_dims, num_node_features, node_embedding_dim, dropout = dropout)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
solver = RGCNSolver(model, lr, n_epochs, device)
solver.fit(train_loader, valid_loader, dev_loader)

# ===================================================== [ Plotting ] ===================================================
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, window_size=10):
    return [np.mean(x[i:i + window_size]) for i in range(len(x) - window_size + 1)]

def plotting_learning_curve(history, window_size=10):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    epochs = np.arange(len(history['train_loss'])) + 1
    ax.plot(epochs[:-window_size + 1], moving_average(history['train_loss'], window_size), label='train_loss')
    ax.plot(epochs[:-window_size + 1], moving_average(history['valid_loss'], window_size), label='valid_loss')
    ax.plot(epochs[:-window_size + 1], moving_average(history['dev_loss'], window_size), label='dev_loss')
    ax.grid()
    ax.legend()

    plt.show(fig)

plotting_learning_curve(solver.history, window_size = 10)

# ====================================================== [ result ] ====================================================
sub = pd.read_csv(f'../data/sample_submission.csv')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model.eval()

preds = []
with torch.no_grad():
    for data in test_loader:
        data.to('cuda')
        pred = model(data.x, data.edge_index, data.edge_type, data.batch).detach().cpu().item()
        sub.loc[sub['uid'] == data.uid[0], 'ST1_GAP(eV)'] = pred

sub.to_csv('sub.csv', index=False)