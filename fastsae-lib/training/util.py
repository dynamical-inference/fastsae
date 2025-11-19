import math
from typing import Optional

from geom_median.torch import compute_geometric_median
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(scheduler_name: Optional[str], optimizer: optim.Optimizer, **kwargs):
    def get_warmup_lambda(warm_up_steps, training_steps):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                return (training_steps - steps) / (training_steps - warm_up_steps)

        return lr_lambda

    # heavily derived from hugging face although copilot helped.
    def get_warmup_cosine_lambda(warm_up_steps, training_steps, lr_end):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                progress = (steps - warm_up_steps) / (training_steps - warm_up_steps)
                return lr_end + 0.5 * (1 - lr_end) * (1 + math.cos(math.pi * progress))

        return lr_lambda

    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name.lower() == "constantwithwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 500)
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / warm_up_steps),
        )
    elif scheduler_name.lower() == "linearwarmupdecay":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        lr_lambda = get_warmup_lambda(warm_up_steps, training_steps)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealing":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps, eta_min=eta_min)
    elif scheduler_name.lower() == "cosineannealingwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        lr_lambda = get_warmup_cosine_lambda(warm_up_steps, training_steps, eta_min)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealingwarmrestarts":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        num_cycles = kwargs.get("num_cycles", 1)
        T_0 = training_steps // num_cycles
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, eta_min=eta_min)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


@torch.no_grad()
def get_init_b_dec(sae, x, init_method, maxiter=100):
    assert init_method == "geometric_median"

    prev_b_dec = sae.b_dec.clone().cpu()
    x = x.cpu()
    median = compute_geometric_median(x, skip_typechecks=True, maxiter=maxiter, per_component=False).median

    median = median.mean(dim=0) if len(median.shape) == 2 else median

    prev_dist = torch.norm(x - prev_b_dec, dim=-1)
    new_dist = torch.norm(x - median, dim=-1)

    print("Reinitializing b_dec with geometric median of activations")
    print(f"Previous distances: {prev_dist.median(0).values.mean().item()}")
    print(f"New distances: {new_dist.median(0).values.mean().item()}")

    new_b_dec = torch.tensor(median, dtype=sae.cfg.dtype, device=sae.cfg.device)
    return new_b_dec
