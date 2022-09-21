
torch.manual_seed(13)

### Build TensorDataset

x_tensor = torch.as_tensor(x_train_tensor).float()
y_tensor = torch.as_tensor(y_train).float()

dataset = TensorDataset(x_tensor, y_tensor)

#Perform the Split
ratio = .8

n_total = len(dataset)
n_train = int(n_total*ratio)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])


#Build train_loader
train_loader = DataLoader(train_data, batch_size = 16, shuffle= True)
val_loader = DataLoader( val_data, 16)
