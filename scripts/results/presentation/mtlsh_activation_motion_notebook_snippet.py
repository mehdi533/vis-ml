from pathlib import Path
import sys

sys.path.insert(0, str(Path("results/thesis_presentation_notebooks").resolve()))

from mtlsh_activation_motion_presentation import (
    DEFAULT_OUTPUT_DIR,
    load_activation_motion_bundle,
    pair_summary,
    plot_activation_dashboard,
    plot_animation_summary,
    plot_binary_comparison,
    plot_sched_change_bars,
    save_current_figure,
    switched_head_neurons,
    switched_shared_neurons,
)


bundle = load_activation_motion_bundle(DEFAULT_OUTPUT_DIR)
pair_summary(bundle)

# Simple editable views:
plot_sched_change_bars(bundle, top_k=10, annotate=True)
plot_binary_comparison(bundle, target="dev_COI", show_pre_activation=False)
plot_animation_summary(bundle)

# Full dashboard:
fig, axes = plot_activation_dashboard(
    bundle,
    target="dev_COI",
    top_k_sched=10,
    show_pre_activation=False,
    figsize=(16, 10),
)

# Useful tables when iterating:
switched_shared_neurons(bundle).head(12)
switched_head_neurons(bundle, target="dev_COI").head(12)

# Optional export:
# save_current_figure("results/thesis_presentation_notebooks/exports/dev_coi_activation_dashboard.png")
