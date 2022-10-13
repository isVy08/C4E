import torch, json, pickle

class Namespace:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

def get_config(config_file): 
  with open(config_file) as f:
        config = json.load(f)
  n = Namespace()
  n.__dict__.update(config)
  return n

def load(datadir):
    with open(datadir, encoding='utf-8') as f:
        data = f.read().splitlines()
    return data

def write(data, savedir, mode='w'):
    f = open(savedir, mode)
    for text in data:
        f.write(text+'\n')
    f.close()


def load_pickle(datadir):
  file = open(datadir, 'rb')
  data = pickle.load(file)
  return data

def write_pickle(data, savedir):
  file = open(savedir, 'wb')
  pickle.dump(data, file)
  file.close()

def load_json(datadir):
    with open(datadir, 'rb') as file:
        return json.load(file)

def write_json(data, savedir):
    with open(savedir, 'w') as file:
        json.dump(data, file, indent=4)

def load_model(model, optimizer, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])