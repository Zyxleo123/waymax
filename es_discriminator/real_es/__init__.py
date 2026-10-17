"""Discriminator trained on recorded Diffusion-ES candidates (safety_progress, full Gaussian WM).

waymax_labels        Waymax-exact overlap/offroad labels for candidates and the logged expert
check_alignment      time/frame consistency of recorded candidates, labels and context
wm_cache             cached world-model predictions and road-edge tokens per replan group
wm_model             WMTrajectoryDiscriminator (full Gaussian WM + road-edge branches)
dataset              in-memory groups, filters and the scenario split
trainer              training loop used by safety_progress
evaluate             offline comparison of discriminator / ES / oracle / random picks
deploy               scorer with the planner's inputs (SimulatorState + world-frame population)
closed_loop          learned ES reward hook used by simulation/run_simulation.py
compare_closed_loop  closed-loop comparison against handcrafted baselines
"""
