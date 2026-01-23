# Data Generation (ANDES TDS → Supervised Dataset)

This part of the repo is used to generate supervised datasets for **Virtual Inertia Scheduling (VIS)** in **IBR-dominated power systems**. It runs **ANDES time-domain simulations (TDS)** on a benchmark grid (IEEE39), applies disturbances (load steps, trips), varies IBR virtual inertia/damping setpoints (M, D), and extracts labels such as **COI frequency nadir**, **RoCoF**, and **IBR power peaks**.

The output is a **flat CSV dataset** that can be used directly for training surrogate models and for end-to-end scheduling evaluation.

Parallel runs are supported via multiprocessing. Set `workers` in `data_generation/generation.yaml` to the number of processes; each worker writes a CSV shard that is merged into the final `output_csv`.
