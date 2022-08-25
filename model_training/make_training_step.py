def make_train_step(model, lossfn, optimizer):
  '''
  Create and Perform a Training Step

  Args: 
  Model - an Instantiated Model
  Lossfn - A loss function
  optimizer = an optimizer

  '''
  def perform_train_step(x,y):
    model.train()
    y_pred =model(x)
    loss = lossfn(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

  return perform_train_step