
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module, accuracy_fn):
  '''
  Return a dictionary containing the model predicting on dataloader.
  '''

  loss = 0
  acc = 0
  with torch.inference_mode():
    for X, y in data_loader:
      y_pred = model(X)

      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_pred, y)

    loss /= len(data_loader)
    acc /= len(data_loader)

  return {'Model_name': model.__class__.__name__, 'model_loss': loss.item(),'model acc': acc.item()}



def train_step(model, dataloader, loss_fn, acccuracy_fn, optimizer):
  '''
  This Funtion takes 
  Args:
  Model 
  Dataloader
  Loss Function
  Accuracy Function
  Optimizer

  returns:
  the loss and Accuracy
  '''
  train_loss = 0
  train_accuracy = 0
  for batch,  (X, y) in enumerate(dataloader):
    model.train()
    X = X.to(device)
    y = y.to(device)
    model.to(device)
    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss
    train_accuracy += accuracy_fn(y_pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  train_loss /= len(dataloader)
  train_accuracy /= len(dataloader)
  print(f'Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f} ')


def test_step(model, dataloader, loss_fn, acccuracy_fn, optimizer):
  test_loss, test_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)

      test_pred = model(X)
      test_loss += loss_fn(test_pred, y)
      test_acc += accuracy_fn(test_pred, y)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}  ')
