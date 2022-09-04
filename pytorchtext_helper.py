def tokenizer(text, how):
  if how == True:
    token = list(text)
  else:
    token = text.split()

  return token
  
