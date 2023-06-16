import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_edge
from utlis import *
from build_graph import build_graph

class GCN(torch.nn.Module):  
    def __init__(self, in_channels, hidden_channels, out_channels,pool_dim):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(in_channels)
        self.conv1 = GCNConv(in_channels, out_channels, cached=True,
                             normalize=True)
        
    def forward(self, x, edge_index, edge_weight, pool_index, emb_matrix):
        x = self.norm(x)
        x = F.dropout(x, p=args.dp1, training=self.training)
        x1 = self.conv1(x, edge_index, edge_weight)
        x2 = F.dropout(x1, p=args.dp2, training=self.training)
        pool_index = F.dropout(pool_index, p=args.dp3, training=self.training)
        x3 = (pool_index.T @ x2) #/ pool_index.sum(axis=0).unsqueeze(1)
        
        return x3

def adjust_learning_rate(optimizer, epoch):
       
    lr = args.lr * (0.8 ** (epoch // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    model.train()
    optimizer.zero_grad()
    x = drop_feature(data.x, args.dropout_rate_feat)
    edge_index, edge_mask = dropout_edge(data.edge_index, p=args.dropout_rate)
    edge_index = edge_index.to(device)
    edge_attr = data.edge_attr[edge_mask].to(device)
    out = model(x, edge_index, edge_attr, data.pool_index, data.emb_matrix)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]) 
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr, data.pool_index, data.emb_matrix).argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


if __name__ == '__main__':
    
    seed_torch(args.seed)
    print(f"==================================================={args.seed}th-train================================================")
    data = build_graph(args.dataset)
    print(data)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    num_classes = torch.unique(data.y).shape[0]
    pool_dim = (data.pool_index.shape[0], data.pool_index.shape[1])
    model = GCN(data.x.shape[1], args.hidden_channels, num_classes,pool_dim)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=0.00001) 


    best_val_acc = final_test_acc = 0
    pre_val_acc = []
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer,epoch)
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        pre_val_acc.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if epoch %50==0:
            print(f"Epoch={epoch}, Loss={loss}, Train={train_acc}, Val={val_acc}, Test={test_acc}")
    
    print(f"===============================================END of {args.seed} with {test_acc}=====================================================")