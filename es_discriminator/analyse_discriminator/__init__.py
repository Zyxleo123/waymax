"""Architecture ablations of the safety_progress WM discriminator (A0 = trained v3 model).

  ablations.py       A0-A4 definitions -> model flags
  test_ablations.py  forward / shape / input-independence tests on synthetic inputs (CPU)
  verify_a0.py       load the A0 checkpoint, check config and that it reproduces its saved test scores
  benchmark.py       trainable parameters and 64-candidate inference latency per ablation
  compare.py         A0-A4 table from each run's eval_test.json (+ benchmark), same test groups checked
  submit_ablations.sh   train A1-A4 with A0's pipeline (slurm/train/train_wm_discriminator.sh --ablation)
  compare_ablations.sh  GPU benchmark of the trained checkpoints + compare.py --paired
"""
