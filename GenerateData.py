from torch.utils.data import DataLoader
from torchvision import datasets






# import torch

# def GenerateData(in_len, out_len, dataset_size):
#     # Generate data
#     addvals = torch.rand(dataset_size)

#     input = []
#     target = []

#     for addval in addvals:
#         input_instance = []
#         target_instance = []

#         curr = torch.tensor(0)
#         input_instance.append(curr)

#         for _ in range(in_len-1):
#             input_instance.append(input_instance[-1] + addval)

#         target_instance.append(input_instance[-1] + addval)
#         for _ in range(out_len-1):
#             target_instance.append(target_instance[-1] + addval)

#         input.append(input_instance)
#         target.append(target_instance)

    
#     return input, target

# test1, test2 = GenerateData(3, 3, 10000)