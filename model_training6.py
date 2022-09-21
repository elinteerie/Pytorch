
n_epochs = 200
losses = []
val_losses = []

for epochs in range(n_epochs):
  loss = mini_batch(device, train_loader, train_step)
  losses.append(loss)


  with torch.inference_mode():
    val_loss = mini_batch(device, val_loader, val_step)
    val_losses.append(val_loss)


  writer.add_scalars(main_tag='loss',
  tag_scalar_dict={'training':loss,
  'validation':val_loss}, global_step= epochs)



writer.close()
