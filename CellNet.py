import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
TODO:
- Adapt the code to work with batched inputs
- Implment cell clusters
- Implement cuda support
- Allow for different core sizes for different cells
"""

class Cell(nn.Module):
    """ Cell """
    def __init__(self, core_size, conn_size, conn_num):
        super().__init__()
        
        # Initialize the core
        core = torch.Tensor(core_size)
        self.core = nn.Parameter(core)
        nn.init.ones_(self.core)

        # Output of the cell (to other cells)
        self.conn_in_core = nn.ModuleList([nn.Sequential(
          nn.Linear(core_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, core_size)
        ) for _ in range(conn_num)])

    def forward(self, x):
        """ Forward """
        # Predicted core adjustment (based on outputs of other cells)
        core_new = self.core
        for idx, module in enumerate(self.conn_in_core):
            pred_grad = module(x[idx])
            core_new = core_new + pred_grad

        # Return the output for other cells and the predicted new core
        return core_new


class InCell(nn.Module):
    """ InCell """
    def __init__(self, core_size, conn_size, conn_num, input_size):
        super().__init__()
        
        # Initialize the core
        core = torch.Tensor(core_size)
        self.core = nn.Parameter(core)
        nn.init.ones_(self.core)

        # Output of the cell (to other cells)
        self.conn_in_core = nn.ModuleList([nn.Sequential(
          nn.Linear(core_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, core_size)
        ) for _ in range(conn_num)])

        # Input to the core
        self.in_to_core = nn.Linear(input_size, core_size)

    def forward(self, x, input=None):
        """ Forward """
        # Predicted core adjustment (based on outputs of other cells)
        core_new = self.core
        for idx, module in enumerate(self.conn_in_core):
            pred_grad = module(x[idx])
            core_new = core_new + pred_grad

        # Adjust predicted core based on input
        if input != None:
            input_comb = self.in_to_core(input)
            core_new = core_new + input_comb

        # Return the output for other cells and the predicted new core
        return core_new
    
class OutCell(nn.Module):
    """ OutCell """
    def __init__(self, core_size, conn_size, conn_num, output_size):
        super().__init__()
        
        # Initialize the core
        core = torch.Tensor(core_size)
        self.core = nn.Parameter(core)
        nn.init.ones_(self.core)

        # Output of the cell (to other cells)
        self.conn_in_core = nn.ModuleList([nn.Sequential(
          nn.Linear(core_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, core_size)
        ) for _ in range(conn_num)])

        # Output of the cell
        self.out_core = nn.Sequential(
          nn.Linear(core_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, output_size)
        )

    def forward(self, x, out=True):
        """ Forward """
        # Predicted core adjustment (based on outputs of other cells)
        core_new = self.core
        for idx, module in enumerate(self.conn_in_core):
            pred_grad = module(x[idx])
            core_new = core_new + pred_grad

        # Adjust predicted core based on input
        if out:
            output = self.out_core(core_new)

        # Return the output for other cells and the predicted new core
        return core_new, output


def temporal_loss(output, target, time=0, alpha=0.1):
    """ Temporal Loss """

    if time < 10:
        loss = ((1+alpha)**time)*F.mse_loss(output, target)
    else:
        loss = ((1+alpha)**10)*F.mse_loss(output, target)

    return loss



        