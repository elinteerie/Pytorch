
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

  return {'Model_name': model.__class__.__name__, 'model_loss': loss.item(),'model acc': acc}


