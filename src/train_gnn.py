"""

GNN Forecast (H2): does a model that forecast (sensor info) performs better than GBM especially during crashes

Model (T-GCN style) for every sensor
1. GCN = each sensor mixes it feature with neighbors 
2. GRU = reads the last hour (12 steps of 5 min) one step at a time
3. GCN again on memory, then a linear layer --> speed change from normal 15 min ahead 
    
Variants
    none          no graph links, each sensor only sees itself 
    freeway       links to the next sensor upstream and downstream (edges.parquet)
    interchange   freeway links + links between freeways (interchange_edges.parquet)

Split: train October-May, June to pick when to stop (validation), test July-September

Outputs:

data/processed/gnn/model_<variant>.pt        the best model (lowest June error)
data/processed/gnn/results_<variant>.csv     test error: all hours, peak, during crashes
    
"""


import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_FILE = "data/processed/gnn/gnn_data.npz"
EDGES_FILE = "data/processed/graph/edges.parquet"
INTER_FILE = "data/processed/graph/interchange_edges.parquet"
OUT_DIR = "data/processed/gnn"

HISTORY = 12             # 1 hour of history (12 x 5 min)
HORIZON = 3              # forecast 15 minutes ahead, same as GBM
VALID_START = "2026-06-01"
TEST_START = "2026-07-01"
TRAIN_EVERY = 2          # use every 2nd time step for training, neighbors 5 min apart are nearly the same
BATCH = 32               # time steps per batch (each one has all 991 sensors)
HIDDEN = 64
MAX_EPOCHS = 20
PATIENCE = 3             # stop after 3 epochs without a better June error
LR = 1e-3
CRASH_WEIGHT = 1 / 0.15  # GBM kept every crash row but only 15% of the others, so crashes counted ~6.7x there too
SEED = 0


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")         # Mac GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def graph_matrix(ids, variant):
    # A_hat = D^-1/2 (A + I) D^-1/2, the standard GCN graph matrix (n x n)
    n = len(ids)
    A = np.eye(n, dtype=np.float32)                       # self loops: every sensor keeps its own value
    if variant != "none":
        col = {s: j for j, s in enumerate(ids)}
        links = pd.read_parquet(EDGES_FILE)[["upstream", "downstream"]].to_numpy().tolist()
        if variant == "interchange":
            links += pd.read_parquet(INTER_FILE)[["sensor_a", "sensor_b"]].to_numpy().tolist()
        for a, b in links:
            if a in col and b in col:
                A[col[a], col[b]] = A[col[b], col[a]] = 1.0   # two way: queues travel upstream, traffic downstream
    d = 1 / np.sqrt(A.sum(axis=1))
    return torch.tensor(d[:, None] * A * d[None, :])     # scaling so busy sensors don't get bigger sums


class TGCN(nn.Module):
    def __init__(self, A_hat, n_in, hidden=HIDDEN):
        super().__init__()
        self.register_buffer("A", A_hat)
        self.gcn1 = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.gcn2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: batch x time x sensors x features
        b, t, n, f = x.shape
        h = torch.relu(self.A @ self.gcn1(x))                  # 1. mix each sensor with its neighbors, every step
        h = h.permute(0, 2, 1, 3).reshape(b * n, t, -1)       # every sensor becomes its own sequence
        _, last = self.gru(h)                                  # 2. GRU memory after the last step
        h = last[-1].reshape(b, n, -1)
        h = torch.relu(self.A @ self.gcn2(h))                  # 3. mix the memories with the neighbors
        # the model starts from "the gap from normal stays the same" and only learns the correction,
        # so during a crash it doesn't have to learn from scratch that slow traffic stays slow
        return x[:, -1, :, 0] + self.out(h).squeeze(-1)       # batch x sensors: change from normal in 15 min


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "freeway"
    assert variant in ("none", "freeway", "interchange"), "variant must be none, freeway or interchange"
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"variant: {variant}, device: {device}")

    # loading the arrays (time x sensors)
    d = np.load(DATA_FILE)
    times = pd.DatetimeIndex(d["times"])
    ids = d["stations"]
    kf = d["speed_kf"].astype(np.float32)
    normal = d["normal"].astype(np.float32)
    std = d["kf_std"].astype(np.float32)
    real = d["real"].astype(np.float32)
    in_crash = d["in_crash"]
    slot, weekend = d["slot"], d["weekend"]

    # input features for every sensor and step, scaled to roughly -1..1 so training is stable
    angle = 2 * np.pi * slot / 288
    per_time = np.stack([np.sin(angle), np.cos(angle), weekend.astype(np.float32)], axis=1)   # same for all sensors
    X = np.stack([(kf - normal) / 10,                     # how far from normal right now
                  (normal - 60) / 10,                     # what normal is at this time
                  np.minimum(std, 20) / 10], axis=2)      # how unsure kalman was
    X = np.concatenate([X, np.broadcast_to(per_time[:, None, :], (*kf.shape, 3))], axis=2).astype(np.float16)  # float16 = half the memory
    Y = (real - normal) / 10                               # target: change from normal, NaN where imputed
    print(f"{len(times):,} steps x {len(ids)} sensors x {X.shape[2]} features")

    # a forecast made at step t uses steps t-11..t and predicts t+3, all steps must be 5 min apart (no missing days)
    gap = np.diff(times.values).astype("timedelta64[m]").astype(int) != 5
    broken = np.concatenate([[0], np.cumsum(gap)])       # changes value after every missing stretch
    t = np.arange(HISTORY - 1, len(times) - HORIZON)
    t = t[broken[t - HISTORY + 1] == broken[t + HORIZON]]
    train_t = t[times[t + HORIZON] < VALID_START][::TRAIN_EVERY]
    valid_t = t[(times[t] >= VALID_START) & (times[t + HORIZON] < TEST_START)]
    test_t = t[times[t] >= TEST_START]
    print(f"forecasts: train {len(train_t):,} | valid {len(valid_t):,} | test {len(test_t):,} (x {len(ids)} sensors)")

    X = torch.tensor(X, device=device)
    Y = torch.tensor(Y, device=device)
    W = torch.tensor(np.where(in_crash, CRASH_WEIGHT, 1.0).astype(np.float16), device=device)   # loss weight per sensor-step
    del kf, std                                            # not needed anymore, frees memory
    steps = torch.arange(-HISTORY + 1, 1, device=device)

    def batch(tt):
        tt = torch.tensor(tt, device=device)
        return X[tt[:, None] + steps].float(), Y[tt + HORIZON]     # inputs: batch x 12 x sensors x features

    model = TGCN(graph_matrix(ids, variant).to(device), X.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    def predict(tt):
        # returns predicted speed (mph) for the forecast steps tt, as numpy
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(tt), BATCH):
                x, _ = batch(tt[i:i + BATCH])
                out.append(model(x).cpu().numpy())
        return np.concatenate(out) * 10 + normal[tt + HORIZON]

    def valid_mae():
        p = predict(valid_t)
        y = real[valid_t + HORIZON]
        m = np.isfinite(y)
        return np.abs(p[m] - y[m]).mean()

    # training: MAE on real readings only, crash steps weighted like in GBM
    best, bad, path = np.inf, 0, os.path.join(OUT_DIR, f"model_{variant}.pt")
    rng = np.random.default_rng(SEED)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(train_t)
        for i in range(0, len(order), BATCH):
            tt = order[i:i + BATCH]
            x, y = batch(tt)
            w = W[torch.tensor(tt, device=device)].float()
            m = ~torch.isnan(y)
            loss = ((model(x)[m] - y[m]).abs() * w[m]).sum() / w[m].sum()   # weighted MAE, real readings only
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            if epoch == 1 and i == 20 * BATCH:
                per = (time.time() - t0) / 21
                print(f"  ~{per * len(order) / BATCH / 60:.1f} min per epoch")
        v = valid_mae()
        print(f"epoch {epoch:2d}  train {total / (len(order) / BATCH) * 10:.2f} mph  June {v:.2f} mph  ({time.time() - t0:.0f}s)")
        if v < best:
            best, bad = v, 0
            torch.save(model.state_dict(), path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("June error stopped improving, stopping")
                break

    # testing on July-September with the best model, same rules as GBM:
    # the sensor has a real reading now and 15 min later
    model.load_state_dict(torch.load(path, map_location=device))
    pred = predict(test_t)
    y = real[test_t + HORIZON]
    now = real[test_t]
    ok = np.isfinite(y) & np.isfinite(now)
    hour = times[test_t].hour.to_numpy()[:, None].repeat(len(ids), axis=1)
    peak = ((hour >= 6) & (hour <= 8)) | ((hour >= 15) & (hour <= 18))
    crash = in_crash[test_t]

    def mae(p, mask):
        mask = mask & ok
        return np.abs(p[mask] - y[mask]).mean()

    every = np.ones_like(ok)
    res = pd.DataFrame({
        "persistence": [mae(now, every), mae(now, peak), mae(now, crash)],
        f"GNN ({variant})": [mae(pred, every), mae(pred, peak), mae(pred, crash)],
    }, index=["all hours", "peak", "during crashes"]).T
    res["n_crash"] = int((crash & ok).sum())
    res.round(3).to_csv(os.path.join(OUT_DIR, f"results_{variant}.csv"))
    print("\nTest error (mph, July-September):")
    print(res.round(2).to_string())


if __name__ == "__main__":
    main()