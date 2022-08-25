device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set Up Learning Rate
LR = 0.1

torch.manual_seed(42)

# Create a model instance
model = nn.Sequential(nn.Linear(1, 1)).to(device)

# Define an Optimizer
optimizer = optim.SGD(model.parameters(), lr=LR)

# Define Loss
lossfn = nn.MSELoss(reduction='mean')

train_step = make_train_step(model, lossfn, optimizer)
