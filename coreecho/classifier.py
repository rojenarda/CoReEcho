import torch

def get_shallow_mlp_head(dim_in, dim_out=1):
    regressor = torch.nn.Sequential(
            torch.nn.BatchNorm1d(dim_in),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(dim_in, dim_in),
            torch.nn.BatchNorm1d(dim_in),
            torch.nn.GELU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(dim_in, dim_out),
            torch.nn.Sigmoid()
        )
    return regressor
