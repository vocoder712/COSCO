"""Diagnose why COSCO reaches 100% test accuracy on BasicMotions 1-shot after only 2 epochs.

Hypothesis: With 1-shot, each class's "centroid" IS the embedding of its single training
sample. The classifier is effectively 1-NN on whatever embedding the network produces.
Even a random (untrained) ResNet may already give a feature space where the 4 test
clusters land nearest to their own training sample, because BasicMotions is a small,
well-separated dataset (40 test samples, 4 classes, accelerometer signatures).

This script trains for 0, 1, 2, and 10 epochs and reports test accuracy each time.
If 0 epochs already hits ~100%, the "training" isn't doing anything — it's all from
the random feature projection + dataset structure.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from Baselines.ResNet import ResNet
from Prototypical_Loss import PrototypicalLoss
from Prototypical_Loss import prototypical_testing as ptest
from SAM import SAM
from utils.load_data import Dataset, load_data


class Args:
    dataset = "BasicMotions"
    shot = 1
    normalize = False


def evaluate(model, Xtr, ytr, Xte, yte, device):
    model.eval()
    with torch.no_grad():
        # train centroids
        x_tr = torch.from_numpy(Xtr).float().to(device).transpose(1, 2)
        _, embed_tr = model(x_tr)
        y_tr_t = torch.from_numpy(ytr).long().squeeze().to(device)
        crit = PrototypicalLoss(flag="neg")
        centroids = crit._compute_class_centroid(y_tr_t, embed_tr)

        # test
        x_te = torch.from_numpy(Xte).float().to(device).transpose(1, 2)
        _, embed_te = model(x_te)
        pred = ptest(embed_te, centroids).cpu().numpy()
    acc = (pred == yte.ravel()).mean()
    return acc


def train(model, loader, device, n_epoch, lr=0.01, rho=0.1):
    crit = PrototypicalLoss(flag="neg")
    opt = SAM(model.parameters(), torch.optim.SGD, lr=lr, momentum=0.9, rho=rho)
    losses = []
    for epoch in range(n_epoch):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze(dim=1)
            opt.zero_grad()
            _, emb = model(inputs.transpose(1, 2))
            loss = crit(emb, labels)
            loss.backward()
            opt.first_step(zero_grad=True)
            _, emb2 = model(inputs.transpose(1, 2).float())
            crit(emb2, labels).backward()
            opt.second_step(zero_grad=True)
            losses.append(loss.item())
    return losses


def main():
    args = Args()
    Xtr, ytr, Xte, yte = load_data(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = Xtr.shape[-1]
    n_classes = len(np.unique(ytr))

    print(f"train {Xtr.shape}, test {Xte.shape}, classes {n_classes}")
    print()

    # Run each setting 5 times with different seeds, like the original code does.
    for n_epoch in [0, 1, 2, 10]:
        accs = []
        first_losses = []
        last_losses = []
        for seed in range(5):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = ResNet(input_size=input_size, nb_classes=n_classes).to(device)
            if n_epoch > 0:
                loader = DataLoader(
                    Dataset(Xtr, ytr), batch_size=1024, shuffle=True, num_workers=0
                )
                losses = train(model, loader, device, n_epoch)
                first_losses.append(losses[0])
                last_losses.append(losses[-1])
            acc = evaluate(model, Xtr, ytr, Xte, yte, device)
            accs.append(acc)
        msg = f"nEpoch={n_epoch:>2d}  acc per seed = {accs}  mean={np.mean(accs):.3f}"
        if first_losses:
            msg += f"  loss first→last avg = {np.mean(first_losses):.3f}→{np.mean(last_losses):.3f}"
        print(msg)


if __name__ == "__main__":
    main()
