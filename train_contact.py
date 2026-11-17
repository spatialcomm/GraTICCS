"""Train GraTICCS on contact channels with 5-fold spatial cross-fitting and write held-out knockout statistics.

python train_contact.py --out G_real_s0 --seed 0
python train_contact.py --out G_crand_b0.1_s0 --seed 0 --spike 0.1 --spike-prefix spike_crand_beta
python train_contact.py --out L_real_s0 --seed 0 --arch local     # matched local model
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from graticcs.data import DATA, Graph, batches, comp_feats, dev, fold_split, load, target_mask, to_dev, train_stage
from graticcs.model import Background, make_lr_branch
from graticcs.paths import RESULTS

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def niche_covariate(xy, fov, Y, radius=50.0):
    """Mean of all readouts over the other cells within `radius` um (same FOV)."""
    from scipy.spatial import cKDTree
    Ym = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
    pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray")
    pairs = pairs[fov[pairs[:, 0]] == fov[pairs[:, 1]]]
    acc = np.zeros_like(Ym); cnt = np.zeros(len(Ym), np.float32)
    np.add.at(acc, pairs[:, 0], Ym[pairs[:, 1]]); np.add.at(acc, pairs[:, 1], Ym[pairs[:, 0]])
    np.add.at(cnt, pairs[:, 0], 1); np.add.at(cnt, pairs[:, 1], 1)
    return acc / np.maximum(cnt, 1)[:, None]


def run_fold(d, G, k, args, out, ctype, ntypes):
    tr, va, te, trm = fold_split(d, k)
    mu, sd = d["Y"][trm].mean(0), d["Y"][trm].std(0) + 1e-6
    Ystd = (d["Y"] - mu) / sd
    nT = Ystd.shape[1]; C = d["L"].shape[1]; graph = args.arch == "graticcs"
    cache = {}

    def sub(fovs):
        key = tuple(fovs)
        if key not in cache:
            g = G.sub(fovs, Ystd); g["comp"] = torch.as_tensor(comp_feats(g, ctype, ntypes)); cache[key] = g
        return to_dev(cache[key])

    ohT = torch.as_tensor(np.eye(ntypes, dtype=np.float32), device=dev)
    ct = torch.as_tensor(ctype, device=dev)
    tt_of = lambda g: ohT[ct[torch.as_tensor(g["nodes"], device=dev)]]
    xin = lambda g: (torch.cat([g["x"], g["comp"], g["spill"]], 1), tt_of(g), g["ei"], g["eattr"])
    nu = Background(xin(sub(tr[:4]))[0].shape[1], ntypes, nT, graph=graph).to(dev)
    lr = make_lr_branch(args.arch, ntypes, C, nT, args.heads).to(dev)
    M = torch.as_tensor(target_mask(d), device=dev)
    bg = lambda g: nu(*xin(g))

    def comm(g, h):
        tt = tt_of(g)
        w = lr.contributions(tt[g["csrc"]], tt[g["cdst"]], h[g["cdst"]].detach(), g["ceattr"], g["s"], g["cdst"], len(g["nodes"]))
        out_ = torch.zeros(len(g["nodes"]), nT, device=dev); out_.index_add_(0, g["cdst"], w @ (lr.V * M))
        return out_, w

    def mse_on(pred):
        def f(fovs):
            se = m = 0
            with torch.no_grad():
                for b in batches(fovs, args.batch_fovs):
                    g = sub(b); se += ((pred(g) - g["y"]) ** 2).sum().item(); m += g["y"].numel()
            return se / m
        return f

    nu.train()
    b1, e1 = train_stage(nu.parameters(), lambda b: (lambda g: F.mse_loss(bg(g)[0], g["y"]))(sub(b)),
                         mse_on(lambda g: bg(g)[0]), tr, va, nu, per=args.batch_fovs)
    nu.eval()

    def step2(b):
        g = sub(b)
        with torch.no_grad(): y0, h = bg(g)
        return F.mse_loss(y0 + comm(g, h)[0], g["y"]) + args.lam * (lr.V * M).norm(dim=1).sum()

    b2, e2 = train_stage(lr.parameters(), step2, mse_on(lambda g: (lambda o: o[0] + comm(g, o[1])[0])(bg(g))), tr, va, lr,
                         per=args.batch_fovs)
    log(f"fold {k}: val mse background {b1:.4f} ({e1} ep), +LR {b2:.4f} ({e2} ep)")
    lr.eval()

    sse0 = np.zeros(nT); sse1 = np.zeros(nT); dsse = np.zeros((C, nT)); nte = 0; per_fov = []; att_e, att_a = [], []
    with torch.no_grad():
        VM = lr.V * M
        for b in batches(te, args.batch_fovs):
            g = sub(b); y0, h = bg(g); dy, w = comm(g, h); yh = y0 + dy; r = g["y"] - yh
            if graph:
                tt = tt_of(g); att_e.append(g["cedge"])
                att_a.append(lr.head_attention(tt[g["csrc"]], tt[g["cdst"]], h[g["cdst"]], g["ceattr"], g["cdst"], len(g["nodes"])).half().cpu().numpy())
            sse0 += ((g["y"] - y0) ** 2).sum(0).cpu().numpy(); sse1 += (r ** 2).sum(0).cpu().numpy(); nte += len(g["nodes"])
            for t in range(nT):
                # removing channel c changes the residual by D_c: dSSE = sum(2 r D + D^2)
                Dt = torch.zeros(len(g["nodes"]), C, device=dev); Dt.index_add_(0, g["cdst"], w * VM[:, t])
                dsse[:, t] += (2 * r[:, t:t + 1] * Dt + Dt ** 2).sum(0).cpu().numpy()
            fv = d["fov"][g["nodes"]]
            for f in np.unique(fv):
                m = torch.as_tensor(fv == f, device=dev)
                per_fov.append([int(f), float(((g["y"][m] - y0[m]) ** 2).sum()), float(((g["y"][m] - yh[m]) ** 2).sum())])

    json.dump(dict(fold=k, arch=args.arch, ntest=int(nte), targets=d["tnames"], sse_nuisance=sse0.tolist(), sse_full=sse1.tolist(),
                   dsse_knockout=dsse.tolist(), per_fov=per_fov), open(out / f"fold{k}_mo.json", "w"))
    if att_e:
        np.savez_compressed(out / f"fold{k}_mo_attention.npz", edge=np.concatenate(att_e), a=np.concatenate(att_a))
    torch.save({"nu": nu.state_dict(), "lr": lr.state_dict()}, out / f"fold{k}_mo.pt")
    log(f"fold {k}: held-out gain of the LR branch {1 - sse1.sum() / sse0.sum():.4%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run name (folder under results/)")
    ap.add_argument("--arch", choices=["graticcs", "local", "local_type", "local_area"], default="graticcs",
                    help="local: equal neighbour weights; local_*: fixed weighting rules")
    ap.add_argument("--heads", type=int, default=16, help="attention heads of the LR branch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spike", help="planted strength beta; reads <spike-prefix><beta>.npz")
    ap.add_argument("--spike-prefix", default="spike_beta")
    ap.add_argument("--pathway-readouts", action="store_true", help="replace RNA programs by pathway/TF activities")
    ap.add_argument("--niche", action="store_true", help="add the 50 um regional readout covariate")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--batch-fovs", type=int, default=4)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(5)))
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = RESULTS / args.out; out.mkdir(parents=True, exist_ok=True)

    d = load("mo"); z = np.load(DATA / "gt_inputs.npz"); ctype = z["ctype"]; ntypes = int(ctype.max()) + 1
    if args.pathway_readouts:
        pw = DATA / "pathway_tf"; names = json.load(open(pw / "meta.json"))["readouts"]
        d["n_prot"] = z["Yp"].shape[1]; d["Y"] = np.c_[z["Yp"], np.load(pw / "Y_pathway_tf.npz")["Y"]].astype(np.float32)
        d["tnames"] = json.load(open(DATA / "gt_channels.json"))["prot_readout"] + names; d["pw_mask"] = np.load(pw / "mask_contact.npy")
    if args.niche:
        d["x"] = np.c_[d["x"], niche_covariate(z["xy"], d["fov"], d["Y"])].astype(np.float32)
    if args.spike is not None:
        d["Y"] = np.load(DATA / f"{args.spike_prefix}{args.spike}.npz")["Y"]
    # receiver-side receptor features belong to the background branch
    first = {}
    for c, chn in enumerate(d["channels"]): first.setdefault(chn["receptor"], c)
    d["x"] = np.c_[d["x"], d["R"][:, sorted(first.values())]].astype(np.float32)
    log("loaded", d["x"].shape, d["Y"].shape, "channels", d["L"].shape[1])
    G = Graph(d)
    for k in args.folds: run_fold(d, G, k, args, out, ctype, ntypes)


if __name__ == "__main__":
    main()
