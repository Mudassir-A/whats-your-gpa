from torch import nn

class GPAPredictor(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(2, 64)
		self.fc2 = nn.Linear(64, 32)
		self.fc3 = nn.Linear(32, 16)
		self.fc4 = nn.Linear(16, 1)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.2)

	def forward(self, x):
		x = self.dropout(self.relu(self.fc1(x)))
		x = self.dropout(self.relu(self.fc2(x)))
		x = self.dropout(self.relu(self.fc3(x)))
		x = self.fc4(x)
		return x
