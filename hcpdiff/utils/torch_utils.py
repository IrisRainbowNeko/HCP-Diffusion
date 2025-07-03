import torch

def invert_func(func, y, x_min=0.0, x_max=1.0, tol=1e-5, max_iter=100):
    """
    y: [B]
    :return: x [B]
    """
    y = y.to(dtype=torch.float32)
    left = torch.full_like(y, x_min)
    right = torch.full_like(y, x_max)

    for _ in range(max_iter):
        mid = (left+right)/2
        val = func(mid)

        too_large = val>y
        too_small = ~too_large

        left = torch.where(too_small, mid, left)
        right = torch.where(too_large, mid, right)

        if torch.all(torch.abs(val-y)<tol):
            break

    return (left+right)/2