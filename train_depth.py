import os
import numpy as np
from model import DFM
from lstm_model import DFM_LSTM
from DataLoader import DataLoader, load_data
import torch
import torch.optim as optim
from loss import Loss
import matplotlib.pyplot as plt
import json
from model_3d import Depth_3d_model


def create_model_dir(model_name):
    current_dir_files = os.listdir()
    if "models" not in current_dir_files:
        os.mkdir("./models")
    models_dir = os.listdir("./models")
    if model_name not in models_dir:
        new_model_path = "./models/" + model_name
        os.mkdir(new_model_path)


def read_log_file(model_name):
    model_path = "./models/" + model_name
    model_files = os.listdir(model_path)
    loss = []
    depth_loss = []
    simse_loss = []
    ssim_loss = []
    lr = []
    if "LogFile.log" in model_files:
        log_file_path = model_path + "/LogFile.log"
        log_file = open(log_file_path, "r")
        log_file_data = log_file.read().split("\n")[:-1]
        for i in range(len(log_file_data)):
            log_file_data[i] = log_file_data[i].split(": ")
            if log_file_data[i][0] == "Loss":
                loss.append(float(log_file_data[i][1]))
            elif log_file_data[i][0] == "Depth Loss":
                depth_loss.append(float(log_file_data[i][1]))
            elif log_file_data[i][0] == "SIMSE Loss":
                simse_loss.append(float(log_file_data[i][1]))
            elif log_file_data[i][0] == "SSIM Loss":
                ssim_loss.append(float(log_file_data[i][1]))
            elif log_file_data[i][0] == "lr":
                lr.append(float(log_file_data[i][1]))
        log_file.close()
    return loss, depth_loss, simse_loss, ssim_loss, lr


def save_model(model, optimizer, save_model_name, losses, depth_loss, simse_loss, ssim_loss, lr):
    history_loss = []
    history_depth_loss = []
    history_simse_loss = []
    history_ssim_loss = []
    history_lr = []
    if load_model:
        history_loss, history_depth_loss, history_simse_loss, history_ssim_loss, history_lr = read_log_file(save_model_name)
    log_file_path = "./models/" + save_model_name + "/LogFile.log"
    log_file = open(log_file_path, "w")

    for i in range(number_of_epochs):
        losses[i] = sum(losses[i]) / len(losses[i])
        depth_loss[i] = sum(depth_loss[i]) / len(depth_loss[i])
        simse_loss[i] = sum(simse_loss[i]) / len(simse_loss[i])
        ssim_loss[i] = sum(ssim_loss[i]) / len(ssim_loss[i])

    losses = history_loss + losses
    depth_loss = history_depth_loss + depth_loss
    simse_loss = history_simse_loss + simse_loss
    ssim_loss = history_ssim_loss + ssim_loss
    lr = history_lr + lr
    for i in range(len(losses)):
        log_file.write("Epoch " + str(i) + "\n")
        log_file.write("Loss: " + str(losses[i]) + "\n")
        log_file.write("Depth Loss: " + str(depth_loss[i]) + "\n")
        log_file.write("SIMSE Loss: " + str(simse_loss[i]) + "\n")
        log_file.write("SSIM Loss: " + str(ssim_loss[i]) + "\n")
        log_file.write("lr: " + str(lr[i]) + "\n")

    log_file.close()

    plt.plot(losses, label="Loss")
    plt.plot(depth_loss, label="Depth Loss")
    plt.plot(simse_loss, label="SIMSE Loss")
    plt.plot(ssim_loss, label="SSIM Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Graph")
    plt.legend(loc="upper right")
    plt.savefig("./models/" + save_model_name + "/Loss_Graph.png")

    torch.save({
        'model_state_dict': model.state_dict(),
    }, "./models/" + save_model_name + "/" + save_model_name + ".pt")

    print("Saved Model Successfully")


def train(model, data, device, number_of_epochs, learning_rate, sequence_length, W_depth, W_simse, W_ssim, save_model_name,
          load_model_name, load_model, training_videos, checkpoint_length):
    create_model_dir(save_model_name)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if load_model:
        if os.path.exists("./models/" + load_model_name + "/" + load_model_name + ".pt"):   
            checkpoint = torch.load("./models/" + load_model_name + "/" + load_model_name + ".pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model " + load_model_name)
        else:
            print("couldn't found model path. Creating a new model instead")
            load_model = False

    model.train()
    losses = []
    depth_loss = []
    simse_loss = []
    ssim_loss = []
    lr = []
    all_losses = []
    for epoch in range(number_of_epochs):
        if epoch % checkpoint_length == 0:
            print("Saving model")
            torch.save({
                'model_state_dict': model.state_dict(),
            }, "./models/" + save_model_name + "/" + save_model_name + ".pt")
        losses.append([]) 
        depth_loss.append([])
        simse_loss.append([])
        ssim_loss.append([])
        print("Starting Epoch Number " + str(epoch))
        for i in range(len(data)):
            print("Start Training For Video Number " + str(training_videos[i]))
            video_loss = []
            video_depth_loss = []
            video_simse_loss = []
            video_ssim_loss = []
            rgb_frames, depth_frames, imu_data, validity_map_frames = data[i]
            for j in range(len(rgb_frames)):
                rgb = rgb_frames[j].to(device)
                imu = imu_data[j].to(device)
                depth_frame = depth_frames[j][0].to(device)
                validity_map = validity_map_frames[j][0].to(device)
                result = model(rgb, imu)
                
                optimizer.zero_grad()

                L_depth, L_ssim, L_SIMSE = Loss(result[0], depth_frame, validity_map)
                loss = W_depth * L_depth + W_simse * L_SIMSE + W_ssim * L_ssim

                all_losses.append((epoch, i, j, loss.item()))

                loss.backward()
                optimizer.step()

                losses[epoch].append(loss.item())
                depth_loss[epoch].append(L_depth.item())
                simse_loss[epoch].append(L_SIMSE.item())
                ssim_loss[epoch].append(L_ssim.item())

                video_loss.append(loss.item())
                video_depth_loss.append(L_depth.item())
                video_simse_loss.append(L_SIMSE.item())
                video_ssim_loss.append(L_ssim.item())

            print("Video " + str(i) + " Loss = " + str(sum(video_loss) / len(video_loss)))
            print("Video " + str(i) + " Depth Loss = " + str(sum(video_depth_loss) / len(video_depth_loss)))
            print("Video " + str(i) + " SIMSE Loss = " + str(sum(video_simse_loss) / len(video_simse_loss)))
            print("Video " + str(i) + " SSIM Loss = " + str(sum(video_ssim_loss) / len(video_ssim_loss)) + "\n")

            for g in optimizer.param_groups:
                lr.append(g['lr'])
        print(f'Avg losses in epoch {epoch} is {sum(losses[epoch])/len(losses[epoch])}')

    save_model(model, optimizer, save_model_name, losses, depth_loss, simse_loss, ssim_loss, lr)


if __name__ == "__main__":
    with open("env_var.json") as f:
        env_var = json.load(f)
    training_videos = env_var["training_videos"]
    rgb_videos_path = env_var["rgb_videos_path"]
    depth_videos_path = env_var["depth_videos_path"]
    gyro_data_path = env_var["gyro_data_path"]

    resize = (320, 240)
    pop_frames = 10

    sequence_length = 2
    W_depth = 0.1
    W_simse = 0.05
    W_ssim = 1

    save_model_name = "model_3D"
    load_model = True
    load_model_name = "model_3D"
    bilinear = True
    learning_rate = 0.001
    number_of_epochs = 10
    checkpoint_length = 5
    shuffle = True

    dataset = DataLoader(rgb_videos_path, depth_videos_path, gyro_data_path, resize, pop_frames, sequence_length)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device:", device)

    # model = DFM_LSTM(sequence_length=2, n_classes=1).to(device)
    model = Depth_3d_model(imu_length=10,
                           encoder_output_channels=256,
                           lstm_layers=3,
                           sequence_length=sequence_length).to(device)

    data = load_data(dataset, training_videos)
    if shuffle:
        data = dataset.shuffle_data()

    if not os.path.exists("./models"):
        os.mkdir("./models")
    if not os.path.exists("./models/" + save_model_name):
        os.mkdir(os.path.join("./models/", save_model_name))
    with open("./models/" + save_model_name + "/training hyper parametes.log", "w") as f:
        f.write(f'epochs: {number_of_epochs}\nW_depth:{W_depth}\nW_simse:{W_simse}\nbilinear:{bilinear}\nTraining on:{training_videos}')
        
    train(model=model,
          data=data,
          device=device,
          number_of_epochs=number_of_epochs,
          learning_rate=learning_rate,
          sequence_length=sequence_length,
          W_depth=W_depth,
          W_simse=W_simse,
          W_ssim=W_ssim,
          save_model_name=save_model_name,
          load_model_name=load_model_name,
          load_model=load_model,
          training_videos=training_videos,
          checkpoint_length=checkpoint_length
          )
