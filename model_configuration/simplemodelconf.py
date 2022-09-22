

lr = 0.1

torch.manual_seed(42)

model = nn.Sequential(nn.Linear(1, 1))


optimizer = optim.SGD(model.parameters(), lr =lr)

loss_fn = nn.MSELoss(reduction='mean')
