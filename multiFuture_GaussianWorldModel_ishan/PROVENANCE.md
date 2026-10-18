# Vendored copy of Ishan Jain's multi-future Gaussian world model

`multiFuture_GaussianWorldModel/` from mini-hong/waymax_rs branch `ishan` at `5c2e2c7ecbac40818275813b9323b0b658020e0a`
(byte-identical, `git archive`). It lives under its own directory so it can be loaded next
to Sagar Bellad's package (es_WM's `multiFuture_GaussianWorldModel/`): `_load_isolated_wm_forecaster`
imports each package under its own alias (`_wmpkg_<dir name>`) with its own `data` modules.
Selected with `--wm_backend multifuture_ishan`. Checkpoint (Ishan, 2026-10-18):
/zfsauton/scratch/ishanj/world_model/ckpts_multifuture_gaussian/multiFuture_GaussianWorldModel_20261005_092950/epoch_0450.pt
