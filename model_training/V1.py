# Define Number of Epochs
n_epochs = 1000

losses = []

for epoch in range(n_epochs):
    loss = train_step(X_train, y_train)
    losses.append(loss)
