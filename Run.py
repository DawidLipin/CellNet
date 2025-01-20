from CellNet import *


cuda = True if torch.cuda.is_available() else False

cellnum = 10
incellnum = 3
outcellnum = 3


cells= []

for _ in range(incellnum):
    cells.append(InCell(64, 64, cellnum, 28))

for _ in range(cellnum - incellnum - outcellnum):
    cells.append(Cell(64, 64, cellnum))

for _ in range(outcellnum):
    cells.append(OutCell(64, 64, cellnum, 28))


if cuda:
    for cell in cells:
        cell.cuda()


optimizers = []

for cell in cells:
    optimizers.append(torch.optim.Adam(cell.parameters()))


