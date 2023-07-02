import torch
import torch.optim as optim
from DataLoader import DataLoader
from model import salt_pepper_filtering_model
from loss import Loss
import matplotlib.pyplot as plt
import os

data_path = "./data"
batch_size = 10
train_percent = 0.8
data_type = "train"
height = 256
width = 512
resize = (width, height)
p = 0.05
input_channels = 1
lr = 0.001
epochs = 10
W_L1 = 0.05
W_ssim = 1
checkpoint_length = 5
save_model_name = "model"
load_model = False
load_model_name = "model"

device = torch.device("cuda")
print("Device:", device)

if not os.path.exists("./models"):
    os.mkdir("./models")
if not os.path.exists("./models/" + save_model_name):
    os.mkdir(os.path.join("./models", save_model_name))

data = DataLoader(data_path=data_path, batch_size=batch_size, train_percent=train_percent, resize=resize,
                  p=p, data_type=data_type)
print("Data length", len(data))

model = salt_pepper_filtering_model(input_channels=input_channels).to(device)
if load_model:
    checkpoint = torch.load("./models/" + load_model_name + "/" + load_model_name + ".pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model " + load_model_name)

optimizer = optim.AdamW(model.parameters(), lr=lr)

losses = []
L1_losses = []
ssim_losses = []
for epoch in range(epochs):
    print("----------------")
    print("Starting epoch number", epoch)
    if epoch % checkpoint_length == 0:
        print("Saving model")
        torch.save({
            'model_state_dict': model.state_dict(),
        }, "./models/" + save_model_name + "/" + save_model_name + ".pt")
    for i in range(len(data)):
        noisy_image, image = data[i]
        noisy_image = noisy_image.to(device)
        image = image.to(device)

        result = model(noisy_image)

        optimizer.zero_grad()

        L1_loss, ssim_loss = Loss(result, image)
        loss = W_L1 * L1_loss + W_ssim * ssim_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        L1_losses.append(L1_loss.item())
        ssim_losses.append(ssim_loss.item())

    print("Average loss: " + "{:.4f}".format(sum(losses) / len(losses)))
    print("Average L1 loss : " + "{:.4f}".format(sum(L1_losses) / len(L1_losses)))
    print("Average SSIM loss: " + "{:.4f}".format(sum(ssim_losses) / len(ssim_losses)))

torch.save({
            'model_state_dict': model.state_dict(),
        }, "./" + save_model_name + ".pt")

plt.plot(losses, label="Loss")
plt.plot(L1_losses, label="Depth Loss")
plt.plot(ssim_losses, label="SSIM Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Graph")
plt.legend(loc="upper right")
plt.savefig("./models/" + save_model_name + "/Loss_Graph.png")


