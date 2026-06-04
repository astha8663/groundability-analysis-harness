import argparse
import glob
import os
import re

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from data_cnn60 import NTUDataLoaders
from model import MLP, Encoder

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def parse_arg():
    p = argparse.ArgumentParser(
        description='Groundability diagnostic for a trained SA-DVAE model')
    p.add_argument('--ss', type=int, required=True)
    p.add_argument('--st', type=str, required=True)
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--dataset_path', type=str, required=True)
    p.add_argument('--wdir', type=str, required=True)
    p.add_argument('--le', type=str, required=True)
    p.add_argument('--ve', type=str, required=True)
    p.add_argument('--tm', type=str, required=True)
    p.add_argument('--num_classes', type=int, required=True)
    p.add_argument('--latent_size', type=int, required=True)
    p.add_argument('--i_latent_size', type=int, required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--load_epoch', type=int, default=None,
                   help="matched se epoch (default: auto-pick the one matching classifier.pth.tar)")
    p.add_argument('--seed', type=int, default=5)
    p.add_argument('--out', type=str, default=None,
                   help="CSV output path (default: <ckpt_dir>/groundability_<ss>.csv)")
    return p.parse_args()


def per_class_groundability(pred_text, true_text, labels, unseen_inds):
    """Mean cosine(predicted text, true class text) per unseen class (absolute)."""
    g = {}
    for c in unseen_inds:
        mask = labels == c
        if mask.sum() == 0:
            g[c] = np.nan
            continue
        cos = cosine_similarity(pred_text[mask], true_text[c][None, :])[:, 0]
        g[c] = float(cos.mean())
    return g


def per_class_groundability_skill(pred_text, true_text, labels, unseen_inds, mean_text):
    """Skill score per unseen class: how much better the probe predicts the true
    class text than a constant 'predict the global mean text' baseline.

    g_c = cos(pred, true) - cos(mean_text, true), averaged over the class's samples.
    A probe with no information predicts ~mean_text, giving g_c ~ 0. This removes the
    text-geometry floor that inflates raw cosine when class embeddings cluster tightly.
    """
    mean_unit = mean_text / np.linalg.norm(mean_text)
    g = {}
    for c in unseen_inds:
        mask = labels == c
        if mask.sum() == 0:
            g[c] = np.nan
            continue
        cos = cosine_similarity(pred_text[mask], true_text[c][None, :])[:, 0]
        baseline = float(np.dot(mean_unit, true_text[c]))
        g[c] = float(cos.mean() - baseline)
    return g


def fit_probe_and_score(X_seen, T_seen, X_unseen, true_text, z_labels, unseen_inds, mean_text):
    scaler = StandardScaler().fit(X_seen)
    ridge = RidgeCV(alphas=ALPHAS).fit(scaler.transform(X_seen), T_seen)
    pred = ridge.predict(scaler.transform(X_unseen))
    return per_class_groundability_skill(pred, true_text, z_labels, unseen_inds, mean_text)


def find_matched_se(ckpt_dir):
    clf_path = f'{ckpt_dir}/classifier.pth.tar'
    clf_mtime = os.path.getmtime(clf_path)
    best_ep = None
    for path in glob.glob(f'{ckpt_dir}/se_*.pth.tar'):
        ep = int(re.search(r'se_(\d+)\.pth\.tar', path).group(1))
        if os.path.getmtime(path) <= clf_mtime + 1:  # written no later than the classifier
            if best_ep is None or ep > best_ep:
                best_ep = ep
    return best_ep


def per_class_error(args, X_unseen, z_labels, unseen_inds, device):
    """SA-DVAE per-unseen-class error from the matched checkpoint."""
    if args.ve in ('shift', 'stgcn'):
        vis = 256
    elif args.ve == 'posec3d':
        vis = 512
    else:
        raise ValueError('Unknown visual embedding model')
    ckpt_dir = f'{args.wdir}/{args.le}/{args.tm}'
    epoch = args.load_epoch if args.load_epoch is not None else find_matched_se(ckpt_dir)
    se_path = f'{ckpt_dir}/se_{epoch}.pth.tar'
    print(f'Using matched sequence encoder: se_{epoch}')

    se = Encoder([vis, args.latent_size + args.i_latent_size], args.i_latent_size).to(device)
    clf = MLP([args.latent_size, args.ss]).to(device)
    se.load_state_dict(torch.load(se_path, weights_only=False)['state_dict'])
    clf.load_state_dict(torch.load(f'{ckpt_dir}/classifier.pth.tar', weights_only=False)['state_dict'])
    se.eval()
    clf.eval()

    u_inds = np.sort(unseen_inds)
    preds = np.empty(len(z_labels), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(X_unseen), args.batch_size):
            xb = torch.from_numpy(X_unseen[i:i + args.batch_size]).float().to(device)
            smu, _ = se(xb)
            out = clf(smu)
            preds[i:i + args.batch_size] = u_inds[torch.argmax(out, -1).cpu().numpy()]

    err = {}
    for c in unseen_inds:
        mask = z_labels == c
        err[c] = float(1.0 - (preds[mask] == c).mean()) if mask.sum() else np.nan
    return err


def main():
    args = parse_arg()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Class text targets (same target SA-DVAE aligns to), L2-normalized
    tfl = [np.load(f'resources/text_feats/{args.dataset}/{args.le}/{m}_{args.num_classes}.npy')
           for m in args.tm.split('_')]
    text = np.concatenate(tfl, axis=-1).astype(np.float64)
    text = text / np.linalg.norm(text, axis=1, keepdims=True)

    unseen_inds = np.sort(
        np.load(f'resources/label_splits/{args.dataset}/{args.st}u{args.ss}.npy'))
    print(f'unseen classes (n={len(unseen_inds)}): {unseen_inds.tolist()}')
    if len(unseen_inds) < 10:
        print('WARNING: <10 unseen classes; Spearman over this few points is not meaningful '
              '(use the 48/12 split, not 55/5).')

    # Skeleton features: train.npy = seen samples, ztest.npy = unseen test samples
    X_seen = np.load(args.dataset_path + '/train.npy').astype(np.float64)
    Y_seen = np.load(args.dataset_path + '/train_label.npy')
    X_unseen = np.load(args.dataset_path + '/ztest.npy').astype(np.float64)
    Y_unseen = np.load(args.dataset_path + '/z_label.npy')
    T_seen = text[Y_seen]
    mean_text = T_seen.mean(axis=0)  # 'no-information' prior: the probe's best constant guess

    # (1) Real groundability (skill score over the mean-text baseline)
    g = fit_probe_and_score(X_seen, T_seen, X_unseen, text, Y_unseen, unseen_inds, mean_text)

    # (2) Hewitt & Liang control: shuffle the seen class -> text assignment, refit
    seen_classes = np.unique(Y_seen)
    perm = seen_classes.copy()
    np.random.shuffle(perm)
    remap = {c: perm[i] for i, c in enumerate(seen_classes)}
    T_seen_shuf = text[np.array([remap[c] for c in Y_seen])]
    g_ctrl = fit_probe_and_score(X_seen, T_seen_shuf, X_unseen, text, Y_unseen, unseen_inds, mean_text)

    # (3) Random-feature floor: probe on Gaussian features of equal dimension
    rng = np.random.default_rng(args.seed)
    Xr_seen = rng.standard_normal(X_seen.shape)
    Xr_unseen = rng.standard_normal(X_unseen.shape)
    g_floor = fit_probe_and_score(Xr_seen, T_seen, Xr_unseen, text, Y_unseen, unseen_inds, mean_text)

    # (4) SA-DVAE per-class error from the matched checkpoint
    err = per_class_error(args, X_unseen, Y_unseen, unseen_inds, device)

    classes = list(unseen_inds)
    gv = np.array([g[c] for c in classes])
    ev = np.array([err[c] for c in classes])
    gc = np.array([g_ctrl[c] for c in classes])
    gf = np.array([g_floor[c] for c in classes])

    rho, pval = spearmanr(gv, ev)
    selectivity = float(np.mean(gv - gc))
    floor = float(np.mean(gf))

    print('\n class | groundability | error')
    print('-------+---------------+-------')
    for c, gi, ei in zip(classes, gv, ev):
        print(f'{c:6d} | {gi:13.4f} | {ei:.4f}')
    print('-------+---------------+-------')
    print(f'\nSpearman rho(groundability, error) = {rho:.4f}  (p={pval:.4f}, n={len(classes)})')
    print(f'selectivity (real - shuffled control) = {selectivity:.4f}  '
          f'[mean real {gv.mean():.4f} vs shuffled {gc.mean():.4f}]')
    print(f'random-feature floor (mean groundability) = {floor:.4f}')
    if selectivity < 0.05:
        print('WARNING: low selectivity; the groundability score may reflect probe capacity, '
              'not skeleton information. Correlation is not interpretable.')

    out_path = args.out or f'{args.wdir}/{args.le}/{args.tm}/groundability_{args.ss}.csv'
    with open(out_path, 'w') as f:
        f.write('class,groundability,error,groundability_shuffled,groundability_floor\n')
        for c, gi, ei, gci, gfi in zip(classes, gv, ev, gc, gf):
            f.write(f'{c},{gi},{ei},{gci},{gfi}\n')
    print(f'\nsaved per-class table to {out_path}')


if __name__ == '__main__':
    main()



'''
run this file 

python groundability.py --num_classes 60 --ss 12 --st r --ve shift --le stsb-bert-large --tm lb --latent_size 160 --i_latent_size 8 --dataset_path "resources/sk_feats/shift_ntu60_12_r/" --wdir "results/shift_ntu60_12_r/" --dataset ntu60
'''