import os

def draw_history(history, save_path=None):
    train_loss = history["train_loss"]
    train_acc = history["train_acc"]
    valid_loss = history["valid_loss"]
    valid_acc = history["valid_acc"]

    plt.subplot(2,1,1)
    plt.plot(train_loss, label="train")
    plt.plot(valid_loss, label="valid")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(train_acc, label="train")
    plt.plot(valid_acc, label="valid")
    plt.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_path, 'train_plot.png'), dpi=300)

def set_device(device_num=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        device += ':{}'.format(device_num)
    return device

def set_save_path(model_name, epochs, batch_size):
    directory = os.path.join('models', f'{model_name}_e{epochs}_bs{batch_size}')
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
    return directory

def get_name_ext(file_path: str) -> tuple[str, str]:
    """
    :param file_path: absolute or relative file path where file is located
    :return: name, extension
    """
    if os.sep in file_path:
        file_name = file_path.split(os.sep)[-1]
    else:
        file_name = file_path
    if os.extsep in file_name:
        name, ext = file_name.rsplit(os.extsep, maxsplit=1)
    else:
        name, ext = file_name, ""
    return name,