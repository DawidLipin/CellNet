import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
TODO:
- Adapt the code to work with batched inputs
- Implment cell clusters
- Implement cuda support
"""

class Cell(nn.Module):
    """ Cell """
    def __init__(self, core_size, conn_size, conn_num):
        super().__init__()
        
        # Initialize the core
        core = torch.Tensor(core_size)
        self.core = nn.Parameter(core)
        nn.init.ones_(self.core)

        # Connect output of other cells to the core
        self.x_to_core = nn.Linear(conn_size, core_size)

        # Output of the core
        self.core_out = nn.Linear(core_size, conn_size)

        # Combine core and output of other cells
        self.combine = nn.Linear(core_size+conn_size, conn_size)

        # Output of the cell (to other cells)
        self.comb_out = nn.ModuleList([nn.Sequential(
          nn.Linear(conn_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, conn_size),
          nn.ReLU()
        ) for _ in range(conn_num)])

    def forward(self, x):
        """ Forward """
        # Predicted core adjustment (based on outputs of other cells)
        core_new = self.core
        for i in x:
            i = F.tanh(self.x_to_core(i))
            core_new = torch.mul(core_new, i)

        # Output information stored in the core
        y = F.relu(self.core_out(core_new))

        # Create output for other cells
        y_out = []
        for idx, comb in enumerate(self.comb_out):
            y = torch.cat([y, x[idx]], dim=0)
            y = F.relu(self.combine(y))
            y_out.append(comb(y))

        # Return the output for other cells and the predicted new core
        return y_out, core_new


class InCell(nn.Module):
    """ Cell """
    def __init__(self, core_size, conn_size, conn_num, input_size):
        super().__init__()
        
        # Initialize the core
        core = torch.Tensor(core_size)
        self.core = nn.Parameter(core)
        nn.init.ones_(self.core)

        # Connect output of other cells to the core
        self.x_to_core = nn.Linear(conn_size, core_size)

        # Output of the core
        self.core_out = nn.Linear(core_size, conn_size)

        # Combine core and output of other cells
        self.combine = nn.Linear(core_size+conn_size, conn_size)

        # Output of the cell (to other cells)
        self.comb_out = nn.ModuleList([nn.Sequential(
          nn.Linear(conn_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, conn_size),
          nn.ReLU()
        ) for _ in range(conn_num)])

        # Input to the core
        self.in_to_core = nn.Linear(input_size, core_size)
        # Include informations from input in the output for other cells
        self.in_combine = nn.Linear(conn_size+input_size, conn_size)


    def forward(self, x, input=None):
        """ Forward """
        # Predicted core adjustment (based on outputs of other cells)
        core_new = self.core
        for i in x:
            i = F.tanh(self.x_to_core(i))
            core_new = torch.mul(core_new, i)

        # Adjust predicted core based on input
        if input != None:
            input_comb = F.tanh(self.in_to_core(input))
            core_new = torch.mul(core_new, input_comb)

        # Output information stored in the core
        y = F.relu(self.core_out(core_new))

        # Create output for other cells
        y_out = []
        for idx, comb in enumerate(self.comb_out):
            y = torch.cat([y, x[idx]], dim=0)
            y = F.relu(self.combine(y))
            y = comb(y)

            # Adjust the output for other cells based on input
            if input != None:
                y = torch.cat([y, input], dim=0)
                y = self.in_combine(y)

            y_out.append(y)

        # Return the output for other cells and the predicted new core
        return y_out, core_new
    
class OutCell(nn.Module):
    """ Cell """
    def __init__(self, core_size, conn_size, conn_num, output_size):
        super().__init__()
        
        # Initialize the core
        core = torch.Tensor(core_size)
        self.core = nn.Parameter(core)
        nn.init.ones_(self.core)

        # Connect output of other cells to the core
        self.x_to_core = nn.Linear(conn_size, core_size)

        # Output of the core
        self.core_out = nn.Linear(core_size, conn_size)

        # Combine core and output of other cells
        self.combine = nn.Linear(core_size+conn_size, conn_size)

        # Output of the cell (to other cells)
        self.comb_out = nn.ModuleList([nn.Sequential(
          nn.Linear(conn_size, conn_size),
          nn.ReLU(),
          nn.Linear(conn_size, conn_size),
          nn.ReLU()
        ) for _ in range(conn_num)])

        # Output of the cell
        self.out_combine = nn.Linear((conn_num*conn_size) + core_size, output_size)

    def forward(self, x, out=False):
        """ Forward """
        # Predicted core adjustment (based on outputs of other cells)
        core_new = self.core
        for i in x:
            i = F.tanh(self.x_to_core(i))
            core_new = torch.mul(core_new, i)

        # Output information stored in the core
        y = F.relu(self.core_out(core_new))

        # Create output for other cells
        y_out_conns = []
        for idx, comb in enumerate(self.comb_out):
            y = torch.cat([y, x[idx]], dim=0)
            y = F.relu(self.combine(y))
            y = comb(y)
            y_out_conns.append(y)

        # Create an output for the network
        if out:
            y_out = torch.cat(y_out_conns, dim=0)
            y_out = torch.cat([y_out, self.core], dim=0)
            y_out = self.out_combine(y_out)

            # Return the output for other cells, the predicted new core and the output for the network
            return y_out_conns, core_new, y_out

        # Return the output for other cells and the predicted new core
        return y_out_conns, core_new, None


def temporal_loss(output, target, time=0, alpha=0.1):
    """ Temporal Loss """

    if time < 10:
        loss = ((1+alpha)**time)*F.mse_loss(output, target)
    else:
        loss = ((1+alpha)**10)*F.mse_loss(output, target)

    return loss



        