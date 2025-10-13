import torch


def compress_matrix(
    A: torch.Tensor, 
    mask: torch.Tensor, 
    force_dim: int = None, 
    allow_larger_dim=None
) -> torch.Tensor:
    
    if A.shape[:2] != mask.shape:
        raise ValueError("First two dimensions of A and mask must match.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor.")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError(
            f"mask must only contain 0s and 1s. dtype: {mask.dtype}. "
            f"Invalid elements found at indices: {((mask != 0) & (mask != 1)).nonzero().tolist()} "  # Get indices of elements not 0 AND not 1
            f"with corresponding values: {mask[((mask != 0) & (mask != 1))].tolist()}. "  # Get the values at those indices
            f"\nOriginal mask (showing up to first 20 elements if large):\n{mask.flatten()[:20]}{'...' if mask.numel() > 20 else ''}"
        )

    S, E = mask.shape
    trailing_dims_shape = A.shape[2:]
    num_trailing_dims = len(trailing_dims_shape)
    device = A.device

    ones_per_column = mask.sum(dim=0)
    X = ones_per_column.max().item() if force_dim is None else force_dim

    if X == 0:
        return torch.empty((0, E, *trailing_dims_shape), dtype=A.dtype, device=device)

    sorted_row_indices_2d = torch.argsort(mask.float(), dim=0, descending=True) 
    view_shape_for_indices = (S, E, *((1,) * num_trailing_dims))
    expanded_indices = sorted_row_indices_2d.view(view_shape_for_indices).expand_as(A)
    A_gathered = torch.gather(A, 0, expanded_indices) 

    if X <= A_gathered.shape[0]:
        B_candidate = A_gathered[:X, ...] 
    elif allow_larger_dim or allow_larger_dim is None:
        if allow_larger_dim is None:
            print(f"[Warning compress_matrix] Target dimension X ({X}) is larger than "
                      f"A's original row count S ({S}). Padding B_candidate with zeros.")
        B_candidate = A_gathered 
        zeros_shape = [X - A_gathered.shape[0]] + list(B_candidate.shape[1:])
        B_candidate = torch.cat((B_candidate, torch.zeros(zeros_shape, dtype=B_candidate.dtype, device=B_candidate.device)), dim=0)  # Shape (X_target_dim, E, ...)
    else:
        raise AssertionError(
                f"Target dimension X ({X}) is larger than A's original row count S ({S}) "
                f"and allow_larger_dim is False. Padding is disallowed."
            )

    row_indices_for_B = torch.arange(X, device=device).unsqueeze(1)  
    b_mask_2d = row_indices_for_B < ones_per_column.unsqueeze(0)  
    view_shape_for_b_mask = (X, E, *((1,) * num_trailing_dims))
    B = B_candidate * b_mask_2d.view(view_shape_for_b_mask).to(A.dtype)

    return B


def decompress_matrix(
    B: torch.Tensor, 
    mask: torch.Tensor, 
    allow_larger_dim=None
) -> torch.Tensor:
    if B.shape[1] != mask.shape[1]:
        raise ValueError("B's second dimension and mask's second dimension (E) must match.")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor.")
    if not ((mask == 0) | (mask == 1)).all(): 
        raise ValueError("mask must only contain 0s and 1s.")

    S, E = mask.shape
    X = B.shape[0]
    trailing_dims_shape = B.shape[2:]
    num_trailing_dims = len(trailing_dims_shape)
    device = B.device

    if X == 0: 
        return torch.zeros((S, E, *trailing_dims_shape), dtype=B.dtype, device=device)
    if X <= S:
        pass
    elif allow_larger_dim or allow_larger_dim is None:
        if allow_larger_dim is None:
                print(f"[Warning decompress_matrix] Input B.shape[0] ({X}) is larger than "
                      f"target A's row count S ({S}). Truncating B to its first {S} rows.")
        B = B[:S, ...]
        X = S
    else:
        raise AssertionError(
                f"Input B.shape[0] ({X}) is larger than target A's row count S ({S}) "
                f"and allow_larger_dim is False. Truncation is disallowed."
            )

    sorted_row_indices_2d = torch.argsort(mask.float(), dim=0, descending=True) 
    target_A_row_indices_2d = sorted_row_indices_2d[:X, :] 
    A_reconstructed = torch.zeros((S, E, *trailing_dims_shape), dtype=B.dtype, device=device)
    view_shape_for_target_indices = (X, E, *((1,) * num_trailing_dims))
    expanded_target_indices = target_A_row_indices_2d.view(view_shape_for_target_indices).expand_as(B)
    A_reconstructed.scatter_(dim=0, index=expanded_target_indices, src=B)

    return A_reconstructed

