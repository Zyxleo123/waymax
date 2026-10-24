# Vendored world models (es_WM)

Byte-identical copies from `git@github.com:mini-hong/waymax_rs.git`, branch
`es_WM`, commit `75e3784` ("clean_up", Sagar Bellad, 2026-10-06). Loaded only
through `rl/diffusion_ft/world_model_forecast.py`, which imports them against the
`data` / `scores` here and restores this repo's afterwards.

| here | upstream path |
|---|---|
| `data/{__init__,preprocess,types,utils}.py` | `multiFuture_GaussianWorldModel/data/` (identical to upstream root `data/`) |
| `scores/polygon.py` | `scores/polygon.py` (identical to ours) |
| `scores/__init__.py` | new, empty (upstream `scores` is a namespace package) |
| `wm_det/{__init__,config,network,inference}.py` | `model/world_model/` (the "production" det/gaussian backend) |
| `wm_mf/{__init__,config,network,planning,inference}.py` | `multiFuture_GaussianWorldModel/model/world_model/` |

Why vendored rather than imported from our `data/`: upstream `_build_map_features`
takes `max_segments=` and returns `(feats, valid, ids)`; ours takes neither and
returns two. The WM's map features must be built exactly as in its training.

Do not edit these files; re-vendor from upstream instead and update the hashes.

```
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ./data/__init__.py
6650f324f79f4c7388823c41da80594b1fbebdb12f7e1abd000c9a28d0e107b5  ./data/preprocess.py
9f9ec47fe9420d867b68ff951d30a928bcbe808c316fbfc07921149109cbf49a  ./data/types.py
938cc19935debf31017bacebd9f735a473807fc1bc3c0585d351aaf89bb4246f  ./data/utils.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ./scores/__init__.py
32d41b5a0b0b43edea83abbcb4c069a815a7cabd27f245c2e439d8cc4d4b6406  ./scores/polygon.py
69020efb0e28c65e1ebbef03f16ac02d3dc3e6252586f33cfbec8c0ec1de61a1  ./wm_det/config.py
1a6a335fbf2af6968bbdcfcb72320f48b19f182fcbf064301969855ab1a3eb27  ./wm_det/inference.py
333a438bd73dc0b80356cc46e738f36687c7cf423cdbf24e90f19768fa600e84  ./wm_det/__init__.py
fb117fa64142bff50467595c3f882594990c97051c664e87cba8d213bf6b9df1  ./wm_det/network.py
fa635998d04eaa4952f280667fb9980b240d5ac80c7b3ba9b877884a5c99d3eb  ./wm_mf/config.py
352108d6597b932bc420e2a7619daa02395a9db24dc5349ede74527d1ce0b129  ./wm_mf/inference.py
1dc44e750838d12ea591eb6075024c1a0d266c6fa17ab8691fc5e2573b044acd  ./wm_mf/__init__.py
9bb708071893c569f8ee7efce1ea9f5434182a352c14afc875e7dde26ca06db8  ./wm_mf/network.py
73bcf71842ebd42cd43642a9b52698562a69c653302cde2be674a8e6d4027190  ./wm_mf/planning.py
```
