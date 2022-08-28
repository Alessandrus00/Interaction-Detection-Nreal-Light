import os
from matplotlib import pyplot as plt

LOGS_PATH = 'logs/epochs/'
EPOCHS = 400
EVAL_PERIOD = 10
MINOVERLAP = 0.5

def plot_map(maps):
    epochs_list = range(EPOCHS+1)
    map_x = []
    for epoch in epochs_list:
        if epoch % EVAL_PERIOD == 0:
            map_x.append(epoch)

    plt.figure()
    plt.plot(map_x, maps, 'red', linewidth = 2, label='Validation mAP')

    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('mAP %s'%str(MINOVERLAP))
    plt.title('mAP Curve')
    plt.legend(loc="upper left")

    plt.savefig(os.path.join(LOGS_PATH, "epoch_map.png"))
    plt.cla()
    plt.close("all")

def plot_loss(loss, val_loss):
    iters = range(len(loss))

    plt.figure()
    plt.plot(iters, loss, 'blue', linewidth = 2, label='Training loss')
    plt.plot(iters, val_loss, 'coral', linewidth = 2, label='Validation loss')

    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(LOGS_PATH, "epoch_loss.png"))

    plt.cla()
    plt.close("all")

def read_values(filename):
    with open(os.path.join(LOGS_PATH, filename)) as f:
        lines = f.read().splitlines()
    values = [float(value) for value in lines]
    return values

if __name__ == "__main__":
    loss = read_values('loss.txt')
    val_loss = read_values('val_loss.txt')
    val_map = read_values('val_map.txt')
    plot_loss(loss, val_loss)
    plot_map(val_map)
