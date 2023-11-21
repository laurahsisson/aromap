import torch


def flatten(mtrx):
    return mtrx.reshape((mtrx.shape[0] * mtrx.shape[1], -1)).squeeze()

def get_idx_grid(width, height, step):
    x_idx, y_idx = torch.meshgrid(torch.arange(start=0,
                                               end=width - 1 + step - 1e-5,
                                               step=step),
                                  torch.arange(start=0,
                                               end=height - 1 + step - 1e-5,
                                               step=step),
                                  indexing='ij')
    grid_idx = torch.stack([x_idx, y_idx], dim=-1)

    assert x_idx.shape == y_idx.shape
    return flatten(grid_idx), x_idx.shape

def assert_shape(shape, expected, argname):
    if not shape == expected:
        raise TypeError(
            f"Expected {argname} with shape of {expected} but got {shape}")


def assert_tensor_shape(tensor, expected, argname):
    assert_shape(tensor.shape, expected, argname)
