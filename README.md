# CellNet

This repository contains the code for building and training a **CellNet**, a novel neural network architecture designed to predict and adjust gradients for its components dynamically.

---

## Network Architecture

The network is composed of multiple interconnected "Cells." These cells form the building blocks of the architecture and come in three types:

1. **Standard Cell**: Interacts only with other cells in the network.
2. **InCell**: Receives external input for training.
3. **OutCell**: Produces outputs that are not connected to other cells.

### Standard Cell Components

Each Standard Cell consists of:
- **Core**: A set of floating-point values treated as inputs and outputs during training.
- **Connections**: Fully connected layers that link the cores of different cells.

**InCells** and **OutCells** include additional connections to handle input and output data, respectively.

---

## How the Network Operates

### Initialization

- All cores are initialized with a value of 1.
- During the first step, all connections are provided with zero vectors to initialize the network. Each cell processes these inputs to generate outputs for the next step.

### Step-by-Step Process

#### **Step 1**
1. Each cell receives inputs from all other cells (via their connections).
   - Example: In a network with cells $A, B, C, D, E$, cell $A$ will receive inputs from $B, C, D, E$, while $B$ will receive inputs from $A, C, D, E$.

   **InCells** receive additional external inputs from the training data.
3. These inputs are used to update the cores of all cells. This is done through the connections within the network, NOT through backpropagation.
4. **OutCells** produce outputs used to calculate a task-specific loss function. This loss generates gradients for all components of the network (cores and connections).
5. Before applying these gradients, the current core values are saved as $C_1$.
6. The network undergoes standard backpropagation, updating the core values. These updated values are saved as $C_2$.
7. Finally, the core values are reset to $C_1$ for the next step.

#### **Subsequent Steps**
For all remaining steps, the process is as follows:

1. Each cell receives inputs from other cells via their connections. **InCells** may optionally receive additional training data.
2. The cores are updated based on the inputs received (not gradients). Save the new core values as $C_1$.
3. Calculate the network loss:
   - **Loss (a):** Measure the difference between the current core values and $C_2$ using Mean Squared Error (MSE).
   - **Loss (b):** If external input is received, calculate an additional loss based on the output.

    **Loss (a)** ensures the network learns to predict and adapt to gradient changes, while **Loss (b)** allows it to learn from the training data. This dual-loss approach enables the network to anticipate and adjust for future data, even in the absence of new inputs.
5. Update the network parameters using standard backpropagation.
6. Save the updated core values as $C_2$.
7. Reset the core values to $C_1$.


## TODO

1. Complete the example notebook (`Example.ipynb`) to demonstrate the functionality of the network.
2. Develop support for cell clustering, enabling networks where cells are not fully connected to every other cell.
3. Add CUDA support to enhance performance on GPU hardware.
