from occm import *
sys.path.append("losses")
from custom_loss import *

x, y = torch.rand(4, 128), torch.rand(4, 2)

c_loss = compactness_loss(x)
d_loss = descriptiveness_loss(y, torch.tensor([0, 1, 0, 1]))

print("Compactness loss:", c_loss)
print("Descriptiveness loss:", d_loss)
