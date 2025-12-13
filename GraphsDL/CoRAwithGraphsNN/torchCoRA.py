import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, BatchNorm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
import numpy as np


os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Planetoid(root="data/Cora", name="Cora")
data = dataset[0].to(device)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 32)
        self.conv2 = GCNConv(32, dataset.num_classes)
        self.dropout = 0.5

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


model = GCN().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)



def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = accuracy_score(
        data.y[mask].cpu(),
        pred[mask].cpu()
    )
    return acc, pred, out

train_losses = []
train_accs = []
val_accs = []

for epoch in range(1, 201):
    loss = train()
    train_acc, _, _ = evaluate(data.train_mask)
    val_acc, _, _ = evaluate(data.val_mask)

    train_losses.append(loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

 



test_acc, test_pred, test_out = evaluate(data.test_mask)
# Classification
y_true = data.y[data.test_mask].cpu().numpy()
y_pred = test_pred[data.test_mask].cpu().numpy()

cls_report = classification_report(y_true, y_pred)
conf_mat = confusion_matrix(y_true, y_pred)


# Loss & Accuracy 
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("outputs/loss_curve.png")
plt.close()

plt.figure()
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("outputs/accuracy_curve.png")
plt.close()


# t-SNE visualize
embeddings = test_out.detach().cpu().numpy()
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(
    emb_2d[:, 0],
    emb_2d[:, 1],
    c=data.y.cpu(),
    cmap="tab10",
    s=10
)
plt.colorbar()
plt.title("GCN Node Embedding Visualization (t-SNE)")
plt.savefig("outputs/tsne_visualization.png")
plt.close()

 
# Report
with open("outputs/analysis_report.txt", "w", encoding="utf-8") as f:
    f.write("GCN on Cora Dataset Analysis Report\n")
    f.write("=" * 40 + "\n\n")

    f.write("Model Architecture : \n")
    f.write("- GCNConv(Features → 32)\n")
    f.write("- GCNConv(32 → Classes)\n\n")

    f.write(f"Final Test Accuaracy：{test_acc:.4f}\n\n")

    f.write("Classification Report：\n")
    f.write(cls_report + "\n")

    f.write("Confusion Matrix：\n")
    f.write(np.array2string(conf_mat))

 
print(f"Test Accuracy: {test_acc:.4f}")
 