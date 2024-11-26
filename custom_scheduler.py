from torch import optim


def get_cosine_warm_up(
    optimizer: optim.Optimizer,
    warm_up_steps: int,
    total_steps: int,
    start_factor: float = 1e-2,
) -> optim.lr_scheduler.LRScheduler:
    """Cosine warm-up scheduler. Called every step

    Args
    ____
    - optimizer (torch.optim.Optimizer)
    - warm_up_steps (int) - number of steps for linear warm-up
    - total_steps (int) - number of total steps in training. Must be greater, than warm_up_steps
    - start_factor (float) - coefficient for optimizer.lr in the begining of warm-up. Default: 1e-2

    Returns
    _______
    torch.optim.lr_scheduler.LRScheduler
    """
    assert (
        warm_up_steps < total_steps
    ), "Total number of steps must be greater then warm-up steps"
    scheduler1 = optim.lr_scheduler.LinearLR(
        optimizer, total_iters=warm_up_steps, start_factor=start_factor
    )

    # cosine lr decay
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(total_steps - warm_up_steps + 2)
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warm_up_steps],  # switch to scheduler2 after warm-up
    )
    return scheduler
