"""# Training Script"""
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import argparse
# model_file is needed to be added
from model_file import network

def create_data_loader(train_data, batch_size):
  return DataLoader(train_data, batch_size=batch_size)

def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
  for input, label in data_loader:
    input, label = input.to(device), label.to(device)
    pred = model(input)
    loss = loss_fn(pred, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f"loss: {loss.item()}")

def Train(model, data_loader, loss_fn, optimizer, epoch, device):
  for i in range(epoch):
    print(f"epoch: {i+1}")
    train_single_epoch(model, data_loader, loss_fn, optimizer, device)
    print("------------------------------")
  print("Training is finished")


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("--data", required=True,
                       help="Your preprocessed data.")
  """ parser.add_argument("--model", required=True,
                       help="Your model.") """
  parser.add_argument("--batch_size", required=True, default=128,
                      help="The batch size.")
  parser.add_argument("--epoch", required=True, default=10,
                      help="The epochs.")
  parser.add_argument("-lr", "--learning_rate", required=True, default=0.001,
                      help="The learning rate.")

  args = parser.parse_args()


  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"

  print(f"device is {device}")

  Batch_size = args.batch_size
  Epoch = args.epoch
  Learning_rate = args.learning_rate

  train_loader = create_data_loader(args.data, Batch_size)

  model_ready = network().to(device)
  print(model_ready)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model_ready.parameters(), lr=Learning_rate)

  Train(model_ready, train_loader, loss_fn, optimizer, Epoch, device)