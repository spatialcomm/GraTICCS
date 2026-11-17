"""Train GraTICCS on secreted channels (all neighbours within 30 um), 5-fold spatial cross-fitting.

python train_secreted.py --out G2_sec_real_s0 --seed 0 --accum 2 --batch-fovs 2
python train_secreted.py --out G2_sec_rand_b0.1_s0 --seed 0 --targets data/breast/spike_sec_rand_beta0.1.npz --accum 2 --batch-fovs 2
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["graticcs", "local", "local_type", "local_dist"], default="graticcs",
                    help="local: equal neighbour weights; local_*: fixed weighting rules")
    ap.add_argument("--heads", type=int, default=16, help="attention heads of the LR branch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", help="npz with a replacement readout matrix Y (planted signals, prior programs)")
    ap.add_argument("--readout-names", help="json with the names of the replacement readouts")
    ap.add_argument("--mask", help="npy channel x readout mask replacing the default one")
    ap.add_argument("--alt-rna-pc", help="npy replacing the receiver RNA principal components")
    ap.add_argument("--pathway-readouts", action="store_true")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--batch-fovs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="mini-batches per optimiser step")
    ap.add_argument("--gpu-mem-frac", type=float, default=0)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(5)))
    args = ap.parse_args()
    if args.gpu_mem_frac > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_frac)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = RESULTS / args.out; out.mkdir(parents=True, exist_ok=True)

    d = load("mo"); z = np.load(DATA / "gt_inputs.npz"); ctype = z["ctype"]; ntypes = int(ctype.max()) + 1
    S = np.load(DATA / "gt_sec_inputs.npz"); smeta = json.load(open(DATA / "gt_sec_channels.json"))
    Ls, Rs = S["L"], S["R"]; chans = smeta["channels"]; C = Ls.shape[1]
    d["Y"] = np.c_[S["Yp"], S["Yr"]].astype(np.float32); d["tnames"] = smeta["prot_readout"] + smeta["rna_programs"]
    n_prot = S["Yp"].shape[1]
    if args.pathway_readouts:
        pw = DATA / "pathway_tf"
        d["Y"] = np.c_[S["Yp"], np.load(pw / "Y_pathway_tf.npz")["Y"]].astype(np.float32)
        d["tnames"] = smeta["prot_readout"] + json.load(open(pw / "meta.json"))["readouts"]
    if args.targets: d["Y"] = np.load(args.targets)["Y"]
    if args.readout_names: d["tnames"] = json.load(open(args.readout_names))["readouts"]
    if args.alt_rna_pc:
        alt = np.load(args.alt_rna_pc); d["x"] = d["x"].copy(); d["x"][:, :alt.shape[1]] = alt
    first = {}
    for k, ch in enumerate(chans): first.setdefault(ch["receptor"], k)
    d["x"] = np.c_[d["x"], Rs[:, sorted(first.values())].astype(np.float32)]
    d["L"] = np.zeros((d["n"], 1), np.float32); d["R"] = np.zeros((d["n"], 1), np.float32)   # contact channels unused here
    G = Graph(d); nT = d["Y"].shape[1]

    M = target_mask({"L": Ls, "tnames": d["tnames"], "channels": chans})
    if args.pathway_readouts: M[:, n_prot:] = np.load(DATA / "pathway_tf" / "mask_secreted.npy")
    if args.mask: M = np.load(args.mask).astype(np.float32)
    ssrc, sdst = S["src"], S["dst"]; seattr = np.c_[np.log1p(S["contact"]), S["dist"] / 15.0].astype(np.float32)
    efov = d["fov"][sdst]; order = np.argsort(efov, kind="stable"); u, st = np.unique(efov[order], return_index=True)
    sec_by_fov = {f: order[a:b] for f, a, b in zip(u, st, np.r_[st[1:], len(order)])}
    log("loaded; channels", C, "readouts", nT, "secreted edges", len(ssrc))

    for k in args.folds:
        torch.manual_seed(args.seed); np.random.seed(args.seed)   # each fold starts from the same seed
        tr, va, te, trm = fold_split(d, k)
        mu, sd = d["Y"][trm].mean(0), d["Y"][trm].std(0) + 1e-6; Ystd = (d["Y"] - mu) / sd
        cache = {}

        def sub(fovs):
            key = tuple(fovs)
            if key not in cache:
                g = G.sub(fovs, Ystd); g["comp"] = torch.as_tensor(comp_feats(g, ctype, ntypes))
                nodes = g["nodes"]; e = np.concatenate([sec_by_fov.get(f, np.empty(0, int)) for f in fovs])
                remap = -np.ones(d["n"], np.int64); remap[nodes] = np.arange(len(nodes))
                g["ssrc"] = torch.as_tensor(remap[ssrc[e]]); g["sdst"] = torch.as_tensor(remap[sdst[e]])
                g["seattr"] = torch.as_tensor(seattr[e]); g["Ln"] = torch.as_tensor(Ls[nodes]); g["Rn"] = torch.as_tensor(Rs[nodes])
                g["tt"] = torch.as_tensor(np.eye(ntypes, dtype=np.float32)[ctype[nodes]]); g["sedge"] = e
                for kk in ("s", "csrc", "cdst", "ceattr", "cedge"): g.pop(kk, None)
                cache[key] = g
            return to_dev(cache[key])

        graph = args.arch == "graticcs"
        xin = lambda g: (torch.cat([g["x"], g["comp"], g["spill"]], 1), g["tt"], g["ei"], g["eattr"])
        nu = Background(xin(sub(tr[:4]))[0].shape[1], ntypes, nT, graph=graph).to(dev)
        lr = make_lr_branch(args.arch, ntypes, C, nT, args.heads).to(dev)
        Mt = torch.as_tensor(M, device=dev)
        bg = lambda g: nu(*xin(g))

        def comm(g, h):
            s = F.softplus(g["Ln"][g["ssrc"]]) * F.softplus(g["Rn"][g["sdst"]])
            w = lr.contributions(g["tt"][g["ssrc"]], g["tt"][g["sdst"]], h[g["sdst"]].detach(), g["seattr"], s, g["sdst"], len(g["nodes"]))
            o = torch.zeros(len(g["nodes"]), nT, device=dev); o.index_add_(0, g["sdst"], w @ (lr.V * Mt))
            return o, w

        def mse_on(pred):
            def f(fovs):
                se = m = 0
                with torch.no_grad():
                    for b in batches(fovs, args.batch_fovs):
                        g = sub(b); se += ((pred(g) - g["y"]) ** 2).sum().item(); m += g["y"].numel()
                return se / m
            return f

        kw = dict(per=args.batch_fovs, accum=args.accum)
        b1, e1 = train_stage(nu.parameters(), lambda b: (lambda g: F.mse_loss(bg(g)[0], g["y"]))(sub(b)), mse_on(lambda g: bg(g)[0]),
                             tr, va, nu, **kw)
        nu.eval()

        def step2(b):
            g = sub(b)
            with torch.no_grad(): y0, h = bg(g)
            return F.mse_loss(y0 + comm(g, h)[0], g["y"]) + args.lam * (lr.V * Mt).norm(dim=1).sum()

        b2, e2 = train_stage(lr.parameters(), step2, mse_on(lambda g: (lambda o: o[0] + comm(g, o[1])[0])(bg(g))), tr, va, lr, **kw)
        lr.eval()

        sse0 = np.zeros(nT); sse1 = np.zeros(nT); dsse = np.zeros((C, nT)); nte = 0; att_e, att_a = [], []
        with torch.no_grad():
            VM = lr.V * Mt
            for b in batches(te, args.batch_fovs):
                g = sub(b); y0, h = bg(g); dy, w = comm(g, h); r = g["y"] - (y0 + dy)
                if graph:
                    att_e.append(g["sedge"])
                    att_a.append(lr.head_attention(g["tt"][g["ssrc"]], g["tt"][g["sdst"]], h[g["sdst"]], g["seattr"], g["sdst"],
                                                   len(g["nodes"])).half().cpu().numpy())
                sse0 += ((g["y"] - y0) ** 2).sum(0).cpu().numpy(); sse1 += (r ** 2).sum(0).cpu().numpy(); nte += len(g["nodes"])
                for t in range(nT):
                    Dt = torch.zeros(len(g["nodes"]), C, device=dev); Dt.index_add_(0, g["sdst"], w * VM[:, t])
                    dsse[:, t] += (2 * r[:, t:t + 1] * Dt + Dt ** 2).sum(0).cpu().numpy()
        if att_e:
            np.savez_compressed(out / f"fold{k}_mo_attention.npz", edge=np.concatenate(att_e), a=np.concatenate(att_a))
        json.dump(dict(fold=k, arch=args.arch, ntest=int(nte), targets=d["tnames"], sse_nuisance=sse0.tolist(), sse_full=sse1.tolist(),
                       dsse_knockout=dsse.tolist()), open(out / f"fold{k}_mo.json", "w"))
        torch.save({"nu": nu.state_dict(), "lr": lr.state_dict()}, out / f"fold{k}_mo.pt")
        log(f"fold {k}: val background {b1:.4f} ({e1}), +LR {b2:.4f} ({e2}); held-out gain {1 - sse1.sum() / sse0.sum():.4%}")


if __name__ == "__main__":
    main()
