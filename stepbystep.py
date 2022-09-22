class StepbyStep(object):
  def __init__(self, model, loss_fn, optimizer):
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    self.train_loader = None
    self.val_loader = None
    self.writer = None


    self.losses = []
    self.val_losses = []
    self.total_epochs = 0

    self.train_step = self.make_train_step()
    self.val_step = self.make_val_step()



  def to(self, device):
    self.device = device
    self.model.to(self.device)



  def set_loaders(self, train_loader, val_loader=None):
    self.train_loader = train_loader
    self.val_loader = val_loader



  def _set_tensorboard(self, name, folder ='runs'):
    suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    self.writer = SummaryWriter('{}/{}_{}'.formtat(folder, 
                                                   name, suffix))
  
  def _make_train_step(self):

    def perform_train_step(X, y):
      self.model.train()
      yhat = self.model(X)
      loss = self.loss_fn(yhat, y)
      loss.backward()
      self.optimizer.step()
      self.optimizer.zero_grad()

      return loss.item()
    return perform_train_step



    def _make_val_step(self):

      def perform_val_step(X,y):

        self.model.eval()

        yhat = model(X)
        loss = self.loss_fn(yhat, y)

        return loss.item()
      return perform_val_step

  def _mini_batch(self, validation = False):

    if validation:
      data_loader = self.val_loader
      step = self.val_step
    else:
      data_loader = self.train_loader
      step = self.train_step


    if data_loader is None:
      return None



    mini_batch_losses = []

    for x_batch, y_batch in data_loader:
      x_batch = x_batch.to(self.device)
      y_batch = y_batch.to(device)


      mini_batch_loss = step(x_batch, y_batch)
      mini_batch_losses.append(mini_batch_loss)

    loss = np.mean(mini_batch_losses)

    return loss

  setattr(StepbyStep, '_mini_batch', _mini_batch)


  def set_seed(self, seed = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    
    np_random.seed(seed)

  setattr(StepbyStep, 'set_seed', set_seed)

  def train(self, n_epochs, seed = 42):
    self.set_seed(42)


    for epoch in range(n_epochs):

      self.total_epochs += 1

      loss = self._mini_batch(validation = False)
      self.losses.append(loss)


      #Validation
      with torch.inference_mode():
        val_loss =  self._mini_batch(validation=True)
        self.val_losses.append(val_loss)


      if self.writer:
        scalars = {'training': loss}
        if val_loss is not None:
          scalars.update({'validation': val_loss})

        self.write.add_scalers(main_tag ='loss', 
                               tag_scaler_dict =scalars,
                               global_step=epoch)
        
    if self.writer:
      self.writer.flush()

  setattr(StepbyStep, 'train', train)

  def save_checkpoint(self, filename):
    checkpoint = checkpoint = {'epoch': self.total_epochs,
                               'model_state_dict': self.model.state_dict(),
                               'optimizer_state_dict': self.optimizer.state_dict(),
                               'loss': self.losses,
                               'val_loss': self.val_losses}
    torch.save(checkpoint, filename )
  setattr(StepbyStep, 'save_checkpoint', save_checkpoint)


  def load_checkpoint(self, filename):
    checkpoint = torch.load(filename)

    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    self.total_epochs = checkpoint['epochs']
    self.losses = checkpoint['loss']
    self.val_losses = checkpoint['val_loss']

    self.model.train()

  setattr(StepbyStep, 'load_checkpoint', load_checkpoint)

  def predict(self, x):
    self.model.eval()
    

    x_tensor = torch.as_tensor(x).float()

    yhat_tensor = self.model(x_tensor.to(self.device))

    self.model.train()

    return yhat_tensor.detach().cpu().numpy()

  setattr(StepbyStep, 'predict', predict)

  def plot_losses(self):
    fig = plt.figure(figsize(10, 4))
    plt.plot(self.losses, label='Training Loss', c = 'b')
    if self.val_loader:
      plt.plot(self.val_losses, label ='Validation Loss', c = 'r')


    plt.yscale('log')
    plt.xlabel('EPochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    return fig
  setattr(StepbyStep, 'plot_losses', plot_losses)

  def add_graph(self):
    if self.train_loader and self.writer:

      x_dummy, y_dummy = next(iter(self.train_loader))
      sef.writer.add_graph(self.model, x_dummy.to(self.device))

  setattr(StepbyStep, 'add_graph', add_graph)









    
  
