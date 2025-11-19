from functools import partial
from itertools import product
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "data/hf_home"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time

import einops
import torch
from transformers import AutoProcessor

from sae.dataset.flexible import create_dataloader
from sae.models import Backbone
from sae.models import SAE
from sae.models import SAEWrapper
from sae.models.wrapper import get_cached_dataloader

# Configuration
backbone_hf_name = "openai/clip-vit-base-patch16"
backbone_hook_target_name = "vision_model.encoder.layers.10"


def sae_wrapper_forward(sae_wrapper, batch):
    z = sae_wrapper.backbone.forward_to_z(batch)
    z_hat, a = sae_wrapper.sae(z)
    return z, z_hat, a


def sae_wrapper_simple_forward(sae_wrapper, batch):
    z = sae_wrapper.backbone.forward_to_z(batch)
    z_hat, a = sae_simple_forward(sae_wrapper.sae, z)
    return z, z_hat, a


def sae_simple_forward(sae, z):
    W_enc = sae.W_enc
    W_dec = sae.W_dec
    b_enc = sae.b_enc
    b_dec = sae.b_dec
    return _sae_simple_forward_pass(W_enc, W_dec, b_enc, b_dec, z)


def _sae_simple_forward_pass(W_enc, W_dec, b_enc, b_dec, z):
    a = _sae_simple_encode(W_enc, b_enc, b_dec, z)
    z_hat = _sae_simple_decode(W_dec, b_dec, a)
    return z_hat, a


def _sae_simple_encode(W_enc, b_enc, b_dec, z):
    # centralize
    z_centralized = z - b_dec
    # encoder
    a_pre = einops.einsum(z_centralized, W_enc, "... d_in, d_in d_sae -> ... d_sae") + b_enc
    return a_pre


def _sae_simple_decode(W_dec, b_dec, a):
    z_hat = einops.einsum(a, W_dec, "... d_sae, d_sae d_in -> ... d_in") + b_dec
    return z_hat


def get_cuda_memory_info():
    """Get current CUDA memory usage in MB"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024  # Convert to MB
        return allocated, reserved
    return 0, 0


def get_models(expansion_ratio=64):
    """Initialize backbone and SAE models"""
    backbone = Backbone(model_hf_name=backbone_hf_name, hook_target_name=backbone_hook_target_name)
    sae = SAE(
        d_in=backbone.hook_target_dim,
        d_sae_factor=expansion_ratio,
        architecture="vanilla",
    )
    return backbone, sae


def get_loader(split="val", batch_size=128, num_workers=8):
    processor = AutoProcessor.from_pretrained(backbone_hf_name)

    def processor_fn(img):
        return processor(images=img, text="a photo", return_tensors="pt")

    train_loader = create_dataloader(
        dataset="imagenet",
        split=split,
        hf_name_map={"imagenet": "evanarlian/imagenet_1k_resized_256"},
        processor=processor_fn,
        return_pixel_values_only=True,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        collate="auto",
    )
    return train_loader


def benchmark_dataloader_speed(loader, label="", max_batches=100):
    print(f"\n=== DATALOADER SPEED {label} ===")

    # Clear memory before benchmark
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    start_time = time.time()
    batch_count = 0

    for _ in loader:
        batch_count += 1
        if batch_count >= max_batches:
            break

    end_time = time.time()
    total_time = end_time - start_time

    # Get memory usage after benchmark
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_allocated_end, mem_reserved_end = get_cuda_memory_info()

    print(f"Processed {batch_count} batches in {total_time:.3f}s")
    print(f"Average: {total_time / batch_count:.3f}s per batch")
    print(f"Speed: {batch_count / total_time:.2f} batches/second")
    print(f"Memory: {mem_allocated_end:.1f}MB allocated, {mem_reserved_end:.1f}MB reserved")
    return total_time, mem_allocated_end, mem_reserved_end


def benchmark_forward_only(forward_fn, loader, label="", max_batches=100):
    print(f"\n=== FORWARD PASS ONLY {label} ===")

    test_batch = next(iter(loader))

    # Clear memory before benchmark
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Warm up
    for _ in range(3):
        with torch.no_grad():
            z, z_hat, a = forward_fn(test_batch)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    start_time = time.time()

    for _ in range(min(max_batches, len(loader))):
        with torch.no_grad():
            z, z_hat, a = forward_fn(test_batch)

    end_time = time.time()
    total_time = end_time - start_time

    # Get memory usage after benchmark
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_allocated_end, mem_reserved_end = get_cuda_memory_info()

    num_passes = min(max_batches, len(loader))
    print(f"Processed {num_passes} forward passes in {total_time:.3f}s")
    print(f"Average: {total_time / num_passes:.3f}s per forward pass")
    print(f"Speed: {num_passes / total_time:.2f} forward passes/second")
    print(f"Memory: {mem_allocated_end:.1f}MB allocated, {mem_reserved_end:.1f}MB reserved")
    return total_time, mem_allocated_end, mem_reserved_end


def benchmark_forward_backward(forward_fn, parameters, loader, label="", max_batches=100):
    """Benchmark forward + backward pass"""
    print(f"\n=== FORWARD + BACKWARD PASS {label} ===")

    test_batch = next(iter(loader))

    optimizer = torch.optim.Adam(parameters, lr=0.0004)

    def loss_fn(out_pre, out_post, a):
        recon_loss = torch.nn.functional.mse_loss(out_post, out_pre)
        sparsity_loss = torch.abs(a).mean()
        return recon_loss + 0.00008 * sparsity_loss

    # Clear memory before benchmark
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Warm up
    for _ in range(3):
        optimizer.zero_grad()
        z, z_hat, a = forward_fn(test_batch)
        loss = loss_fn(z, z_hat, a)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Benchmark
    start_time = time.time()

    for _ in range(min(max_batches, len(loader))):
        optimizer.zero_grad()
        z, z_hat, a = forward_fn(test_batch)
        loss = loss_fn(z, z_hat, a)
        loss.backward()
        optimizer.step()

    end_time = time.time()
    total_time = end_time - start_time

    # Get memory usage after benchmark
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_allocated_end, mem_reserved_end = get_cuda_memory_info()

    num_passes = min(max_batches, len(loader))
    print(f"Processed {num_passes} forward+backward passes in {total_time:.3f}s")
    print(f"Average: {total_time / num_passes:.3f}s per forward+backward pass")
    print(f"Speed: {num_passes / total_time:.2f} forward+backward passes/second")
    print(f"Memory: {mem_allocated_end:.1f}MB allocated, {mem_reserved_end:.1f}MB reserved")
    return total_time, mem_allocated_end, mem_reserved_end


def benchmark_full_training(forward_fn, parameters, loader, label="", max_batches=100):
    print(f"\n=== FULL TRAINING {label} ===")

    optimizer = torch.optim.Adam(parameters, lr=0.0004)

    def loss_fn(out_pre, out_post, a):
        recon_loss = torch.nn.functional.mse_loss(out_post, out_pre)
        sparsity_loss = torch.abs(a).mean()
        return recon_loss + 0.00008 * sparsity_loss

    # Clear memory before benchmark
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Warm up
    warmup_batches = min(3, len(loader))
    for i, batch in enumerate(loader):
        if i >= warmup_batches:
            break
        optimizer.zero_grad()
        z, z_hat, a = forward_fn(batch)
        loss = loss_fn(z, z_hat, a)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Benchmark
    start_time = time.time()
    batch_count = 0

    for batch in loader:
        optimizer.zero_grad()
        z, z_hat, a = forward_fn(batch)
        loss = loss_fn(z, z_hat, a)
        loss.backward()
        optimizer.step()
        batch_count += 1
        if batch_count >= max_batches:
            break

    end_time = time.time()
    total_time = end_time - start_time

    # Get memory usage after benchmark
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_allocated_end, mem_reserved_end = get_cuda_memory_info()

    print(f"Processed {batch_count} training batches in {total_time:.3f}s")
    print(f"Average: {total_time / batch_count:.3f}s per training batch")
    print(f"Speed: {batch_count / total_time:.2f} training batches/second")
    print(f"Memory: {mem_allocated_end:.1f}MB allocated, {mem_reserved_end:.1f}MB reserved")
    return total_time, mem_allocated_end, mem_reserved_end


def main(
    max_batches=[100],
    batch_size=[128],
    expansion_ratio=[64],
    num_workers=[8],
    output_file="benchmark_results.csv",
    overwrite=False,
):
    """Run all benchmarks with CUDA memory tracking"""
    print("Initializing models and dataloaders...")
    print("CUDA memory usage will be tracked for all benchmarks")

    # Collect results
    import os

    import pandas as pd

    # Load existing results if file exists
    existing_results = []
    if os.path.exists(output_file) and not overwrite:
        try:
            existing_df = pd.read_csv(output_file)
            existing_results = existing_df.to_dict("records")
            print(f"Loaded {len(existing_results)} existing results from {output_file}")
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_file}: {e}")
            existing_results = []

    # Track results for this run
    new_results = []

    def append_results_to_csv(config_results, output_file):
        """Append results for a single configuration to CSV file"""
        if not config_results:
            return

        # Create DataFrame for this configuration's results
        df_config = pd.DataFrame(config_results)

        # Append to CSV file
        if os.path.exists(output_file):
            df_config.to_csv(output_file, mode="a", header=False, index=False)
        else:
            df_config.to_csv(output_file, mode="w", header=True, index=False)

        print(f"  → Saved {len(config_results)} results to {output_file}")

    print("\n" + "=" * 80)
    print("RUNNING BENCHMARKS")
    print("=" * 80)

    # Generate all parameter combinations
    param_combinations = list(product(batch_size, expansion_ratio, num_workers, max_batches))
    total_combinations = len(param_combinations)

    print(f"Testing {total_combinations} parameter combinations...")
    print(f"Parameters: batch_size={batch_size}, expansion_ratio={expansion_ratio}, num_workers={num_workers}, max_batches={max_batches}")

    for i, (bs, exp_ratio, nw, max_batch) in enumerate(param_combinations, 1):
        print(f"\n{'=' * 60}")
        print(f"TESTING [{i}/{total_combinations}]: batch_size={bs}, expansion_ratio={exp_ratio}, num_workers={nw}, max_batches={max_batch}")
        print(f"{'=' * 60}")

        # Check if this configuration already exists
        config_key = {
            "batch_size": bs,
            "expansion_ratio": exp_ratio,
            "num_workers": nw,
            "max_batches": max_batch,
        }

        # Check if we already have results for this configuration
        existing_configs = [r for r in existing_results if all(r.get(k) == v for k, v in config_key.items())]

        if existing_configs and not overwrite:
            print(f"Skipping configuration (already exists in {output_file})")
            continue

        # Get models and dataloaders for this configuration
        backbone, sae = get_models(expansion_ratio=exp_ratio)
        train_loader = get_loader(batch_size=bs, split="val", num_workers=nw)

        # Create cached dataloader
        cached_train_loader = get_cached_dataloader(
            dataloader=train_loader,
            backbone=backbone,
            batch_size=2048,
            writer_batch_size=2048,
        )

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # Create wrappers
        wrapper_with_cache = SAEWrapper(
            backbone=backbone,
            sae=sae,
            _caching=True,
        )

        wrapper_without_cache = SAEWrapper(
            backbone=backbone,
            sae=sae,
            _caching=False,
        )

        # Define loaders and wrappers
        loaders = {
            "with_cache": (cached_train_loader, wrapper_with_cache),
            "without_cache": (train_loader, wrapper_without_cache),
        }

        # Define forward functions
        forward_fns = {
            "wrapper_forward": sae_wrapper_forward,
            "simple_forward": sae_wrapper_simple_forward,
        }

        # Collect results for this configuration
        config_results = []

        for loader_name, (loader, wrapper) in loaders.items():
            print(f"\n--- Testing {loader_name.upper()} ---")

            # Dataloader speed (doesn't need wrapper)
            time_dl, mem_allocated_dl, mem_reserved_dl = benchmark_dataloader_speed(loader, f"({loader_name.upper()})", max_batch)
            config_results.append(
                {
                    "batch_size": bs,
                    "expansion_ratio": exp_ratio,
                    "num_workers": nw,
                    "max_batches": max_batch,
                    "loader": loader_name,
                    "benchmark": "dataloader_speed",
                    "time": time_dl,
                    "batches_per_sec": max_batch / time_dl,
                    "forward_fn": "none",
                    "mem_allocated_mb": mem_allocated_dl,
                    "mem_reserved_mb": mem_reserved_dl,
                }
            )

            # Test both forward function types
            for forward_name, forward_fn in forward_fns.items():
                print(f"\n--- Testing {forward_name.upper()} with {loader_name.upper()} ---")

                bound_forward_fn = partial(forward_fn, wrapper)

                # Forward only
                time_fwd, mem_allocated_fwd, mem_reserved_fwd = benchmark_forward_only(
                    bound_forward_fn,
                    loader,
                    f"({loader_name.upper()}, {forward_name.upper()})",
                    max_batch,
                )
                actual_batches = min(max_batch, len(loader))
                config_results.append(
                    {
                        "batch_size": bs,
                        "expansion_ratio": exp_ratio,
                        "num_workers": nw,
                        "max_batches": max_batch,
                        "loader": loader_name,
                        "forward_fn": forward_name,
                        "benchmark": "forward_only",
                        "time": time_fwd,
                        "batches_per_sec": actual_batches / time_fwd,
                        "mem_allocated_mb": mem_allocated_fwd,
                        "mem_reserved_mb": mem_reserved_fwd,
                    }
                )

                # Forward + backward
                time_fwd_bwd, mem_allocated_bwd, mem_reserved_bwd = benchmark_forward_backward(
                    bound_forward_fn,
                    wrapper.sae.parameters(),
                    loader,
                    f"({loader_name.upper()}, {forward_name.upper()})",
                    max_batch,
                )
                config_results.append(
                    {
                        "batch_size": bs,
                        "expansion_ratio": exp_ratio,
                        "num_workers": nw,
                        "max_batches": max_batch,
                        "loader": loader_name,
                        "forward_fn": forward_name,
                        "benchmark": "forward_backward",
                        "time": time_fwd_bwd,
                        "batches_per_sec": actual_batches / time_fwd_bwd,
                        "mem_allocated_mb": mem_allocated_bwd,
                        "mem_reserved_mb": mem_reserved_bwd,
                    }
                )

                # Full training
                time_train, mem_allocated_train, mem_reserved_train = benchmark_full_training(
                    bound_forward_fn,
                    wrapper.sae.parameters(),
                    loader,
                    f"({loader_name.upper()}, {forward_name.upper()})",
                    max_batch,
                )
                config_results.append(
                    {
                        "batch_size": bs,
                        "expansion_ratio": exp_ratio,
                        "num_workers": nw,
                        "max_batches": max_batch,
                        "loader": loader_name,
                        "forward_fn": forward_name,
                        "benchmark": "full_training",
                        "time": time_train,
                        "batches_per_sec": max_batch / time_train,
                        "mem_allocated_mb": mem_allocated_train,
                        "mem_reserved_mb": mem_reserved_train,
                    }
                )

        # Save results for this configuration immediately
        append_results_to_csv(config_results, output_file)
        new_results.extend(config_results)

    # Summary of results
    if new_results:
        print(f"\nCompleted {len(new_results) // 8} unique configurations (8 benchmarks per config)")
        print(f"Total new results: {len(new_results)}")

        # Load all results for analysis
        try:
            all_df = pd.read_csv(output_file)
            print(f"Total results in file: {len(all_df)}")
        except Exception as e:
            print(f"Warning: Could not read final results from {output_file}: {e}")
    else:
        print("No new results generated")

    # Load and filter results for current parameters
    try:
        all_df = pd.read_csv(output_file)

        # Filter results to match current parameters
        filtered_results = all_df[
            (all_df["batch_size"].isin(batch_size))
            & (all_df["expansion_ratio"].isin(expansion_ratio))
            & (all_df["num_workers"].isin(num_workers))
            & (all_df["max_batches"].isin(max_batches))
        ]

        if len(filtered_results) == 0:
            print("No results found matching current parameters")
            return

        print(f"\nFound {len(filtered_results)} results matching current parameters")
        if not new_results:
            print("(Displaying existing results from CSV file)")

    except Exception as e:
        print(f"Warning: Could not load results from {output_file}: {e}")
        if new_results:
            filtered_results = pd.DataFrame(new_results)
        else:
            print("No results to display")
            return

    # Create DataFrame and calculate speedups for display
    df = filtered_results

    # Pivot to get with_cache and without_cache as columns, grouped by all dimensions
    try:
        df_pivot = df.pivot_table(
            index=[
                "benchmark",
                "forward_fn",
                "batch_size",
                "expansion_ratio",
                "num_workers",
                "max_batches",
            ],
            columns="loader",
            values=["time", "batches_per_sec", "mem_allocated_mb", "mem_reserved_mb"],
            aggfunc="mean",
        )
    except Exception as e:
        print(f"Warning: Could not create pivot table: {e}")
        print("Displaying raw results instead:")
        print(df)
        return

    # Calculate speedups (only if both cache configurations exist)
    if ("time", "with_cache") in df_pivot.columns and (
        "time",
        "without_cache",
    ) in df_pivot.columns:
        df_pivot[("speedup", "time_ratio")] = df_pivot[("time", "without_cache")] / df_pivot[("time", "with_cache")]
    if ("batches_per_sec", "with_cache") in df_pivot.columns and (
        "batches_per_sec",
        "without_cache",
    ) in df_pivot.columns:
        df_pivot[("speedup", "throughput_ratio")] = df_pivot[("batches_per_sec", "with_cache")] / df_pivot[("batches_per_sec", "without_cache")]

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Check if we have both cache configurations for meaningful comparison
    available_loaders = set()
    if hasattr(df_pivot, "columns"):
        for col in df_pivot.columns:
            if isinstance(col, tuple) and len(col) == 2:
                available_loaders.add(col[1])

    if "with_cache" not in available_loaders or "without_cache" not in available_loaders:
        print("Note: Cache comparison not available - need both 'with_cache' and 'without_cache' configurations")
        if available_loaders:
            print(f"Available configurations: {list(available_loaders)}")

    # Format the dataframe for better display
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("\nExecution Times (seconds):")
    if "time" in df_pivot.columns:
        time_df = df_pivot["time"].copy()
        if ("speedup", "time_ratio") in df_pivot.columns:
            time_df["speedup"] = df_pivot[("speedup", "time_ratio")]
        print(time_df.round(3))
    else:
        print("No time data available")

    print("\nThroughput (batches/second):")
    if "batches_per_sec" in df_pivot.columns:
        throughput_df = df_pivot["batches_per_sec"].copy()
        if ("speedup", "throughput_ratio") in df_pivot.columns:
            throughput_df["cache_advantage"] = df_pivot[("speedup", "throughput_ratio")]
        print(throughput_df.round(2))
    else:
        print("No throughput data available")

    print("\nMemory Usage (MB):")
    if "mem_allocated_mb" in df_pivot.columns:
        memory_df = df_pivot["mem_allocated_mb"].copy()
        if ("mem_reserved_mb", "with_cache") in df_pivot.columns:
            memory_df["reserved_with_cache"] = df_pivot[("mem_reserved_mb", "with_cache")]
        if ("mem_reserved_mb", "without_cache") in df_pivot.columns:
            memory_df["reserved_without_cache"] = df_pivot[("mem_reserved_mb", "without_cache")]
        print(memory_df.round(1))
    else:
        print("No memory data available")

    print("\nSpeedup Summary:")
    speedup_data = {}
    if ("speedup", "time_ratio") in df_pivot.columns:
        speedup_data["Time Speedup (x)"] = df_pivot[("speedup", "time_ratio")]
    if ("speedup", "throughput_ratio") in df_pivot.columns:
        speedup_data["Throughput Advantage (x)"] = df_pivot[("speedup", "throughput_ratio")]

    if speedup_data:
        speedup_summary = pd.DataFrame(speedup_data)
        print(speedup_summary.round(2))
    else:
        print("No speedup data available (need both cache configurations)")

    print("\nMemory Usage Summary:")
    memory_data = {}
    if ("mem_allocated_mb", "with_cache") in df_pivot.columns:
        memory_data["Allocated (with_cache)"] = df_pivot[("mem_allocated_mb", "with_cache")]
    if ("mem_allocated_mb", "without_cache") in df_pivot.columns:
        memory_data["Allocated (without_cache)"] = df_pivot[("mem_allocated_mb", "without_cache")]
    if ("mem_reserved_mb", "with_cache") in df_pivot.columns:
        memory_data["Reserved (with_cache)"] = df_pivot[("mem_reserved_mb", "with_cache")]
    if ("mem_reserved_mb", "without_cache") in df_pivot.columns:
        memory_data["Reserved (without_cache)"] = df_pivot[("mem_reserved_mb", "without_cache")]

    if memory_data:
        memory_summary = pd.DataFrame(memory_data)
        print(memory_summary.round(1))
    else:
        print("No memory data available")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark SAE performance with and without caching (includes CUDA memory tracking)",
        epilog="""
Examples:
  # Test multiple configurations (4x4x2x2 = 64 combinations)
  python test_caching.py --batch_size 64 128 256 512 --expansion_ratio 8 16 32 64 --num_workers 4 8 --batches 50 100

  # Simple test with single values
  python test_caching.py --batch_size 128 --expansion_ratio 64 --batches 50

  # Custom output file (results saved incrementally, safe for Ctrl+C)
  python test_caching.py --batch_size 64 128 --num_workers 4 8 --output_file results.csv

  # Force recomputation
  python test_caching.py --overwrite --batch_size 64 --expansion_ratio 32

  # Resume interrupted benchmark (skips already completed configurations)
  python test_caching.py --batch_size 64 128 256 --expansion_ratio 8 16 32 64
        """,
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[100],
        help="Maximum number of batches per test (can specify multiple)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Batch size(s) to test (can specify multiple)",
    )
    parser.add_argument(
        "--expansion_ratio",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Expansion ratio(s) to test (can specify multiple)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        nargs="+",
        default=[0, 2, 8, 16],
        help="Number of workers for dataloader (can specify multiple)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="benchmark_results.csv",
        help="CSV file to store results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results in CSV file",
    )

    args = parser.parse_args()
    main(
        max_batches=args.batches,
        batch_size=args.batch_size,
        expansion_ratio=args.expansion_ratio,
        num_workers=args.num_workers,
        output_file=args.output_file,
        overwrite=args.overwrite,
    )
