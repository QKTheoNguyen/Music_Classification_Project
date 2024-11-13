import torch
import torchaudio
import os
import yaml
import datetime
import argparse
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
# tensorboard --logdir "C:\Users\quang\Desktop\Deep Learning Project"
from torch.utils.tensorboard import SummaryWriter
from data import MusicDataset
from model import CNN_Network
from callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train(model, train_loader, valid_loader, config, loss_fn, optimizer, device, epochs, verbose=False):

    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/" + date_time
    if verbose:
        print(f"Training {date_time} started")
    
    # with open(os.path.join(log_dir, "config.yaml"), "w") as file:
    #     yaml.dump(config, file)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_single_epoch(model, train_loader, loss_fn, optimizer, device, verbose)
        valid_loss, valid_accuracy = validate_single_epoch(model, valid_loader, loss_fn, device, verbose)

        print(f"Train Loss: {train_loss}, Validation Loss: {valid_loss}, Validation Accuracy: {valid_accuracy}")

        if epoch == 0:
            early_stopping = EarlyStopping(patience=5)
            reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2)
            tensorboard = TensorBoard(log_dir=log_dir, config=config)
            if not os.path.exists(f"trained/{date_time}"):
                os.makedirs(f"trained/{date_time}")
            checkpoint = ModelCheckpoint(model, save_path=f"trained/{date_time}/model.pth")

        early_stopping(valid_loss)
        reduce_lr(valid_loss, optimizer)
        tensorboard(epoch, train_loss, valid_loss)
        checkpoint(valid_loss)

        if reduce_lr.reduce_lr:
            print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")

        if early_stopping.early_stop:
            print("Early stopping")
            break

    tensorboard.close()

    print("Training completed")

def train_single_epoch(model, train_loader, loss_fn, optimizer, device, verbose=False):
    
    for (data, target) in tqdm(train_loader, leave=False, ncols=80):

        if verbose:
            print(f'data size: {data.size()}')

        data = data.to(device)
        target = target.to(device)
        
        prediction = model(data)
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(f"Loss: {loss.item()}")
    return loss.item()

def validate_single_epoch(model, valid_loader, loss_fn, device, verbose=False):

    # model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for (data, target) in valid_loader:

            data = data.to(device)
            target = target.to(device)

            prediction = model(data)
            if verbose:
                print(f'prediction: {prediction}')
                print(f'target: {target}')

            # import pdb; pdb.set_trace()

            loss = loss_fn(prediction, target)
            # print(f'loss: {loss.item()}')
            val_loss += loss.item()

            _, predicted = torch.max(prediction, 1)
            if verbose:
                print(f'predicted final: {predicted}')
                print(f'valid predictions {predicted == target}')

            total += target.size(0)
            correct += (predicted == target).sum().item()

            if verbose:
                print(f'correct: {correct}')
                print(f'total: {total}')



    accuracy = correct / total
    val_loss /= len(valid_loader)
    # print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}")
    return val_loss, accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/test_config.yaml")
    parser.add_argument("-e","--epochs", type=int, default=None)
    args = parser.parse_args()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = args.config
    config = load_config(config_path)

    metadata_file_train = config["metadata_file_train"]
    metadata_file_valid = config["metadata_file_valid"]
    data_dir = config["data_dir"]

    target_sr = config["target_sr"]
    duration = config["duration"]
    batch_size = config["batch_size"]
    epochs = args.epochs if args.epochs is not None else config["epochs"]

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr,
                                                           n_fft=1024,
                                                           hop_length=512,
                                                           n_mels=64,
                                                           normalized=True)
    
    loss_fn = nn.CrossEntropyLoss()
    model = CNN_Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = MusicDataset(metadata_file_train, data_dir, mel_spectrogram, target_sr, duration, device)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
    
    valid_dataset = MusicDataset(metadata_file_valid, data_dir, mel_spectrogram, target_sr, duration, device)
    valid_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=batch_size, 
                              shuffle=False)
    
    train(model, train_loader, valid_loader, loss_fn, optimizer, device, epochs)
    
    # #### testing ####
    # model_path_v = "trained/20241022-125744/model.pth"
    # model_v = CNN_Network().to(device)
    # model_v.load_state_dict(torch.load(model_path_v))

    # validate_single_epoch(model_v, valid_loader, loss_fn, device)
