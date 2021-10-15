from tensorflow.keras.datasets import cifar10
from image import *
from hashlib import md5


def get_data(param):
    if param["dataset"] == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype(np.float) / 255.
        x_test = x_test.astype(np.float) / 255.

    if param["dataset"] == "GTSRB":
        pass

    if param["dataset"] == "ImageNet":
        pass

    if param["dataset"] == "PubFig":
        pass

    return x_train, y_train, x_test, y_test


def poison(x_train, y_train, param):
    target_label = param["target_label"]
    num_images = int(param["poisoning_rate"] * y_train.shape[0])

    index = np.where(y_train != target_label)
    index = index[0]
    index = index[:num_images]
    x_train[index] = poison_frequency(x_train[index], y_train[index], param)
    y_train[index] = target_label
    return x_train


def poison_frequency(x_train, y_train, param):
    if x_train.shape[0] == 0:
        return x_train

    x_train *= 255.
    if param["YUV"]:
        x_train = RGB2YUV(x_train)

    # transfer to frequency domain
    x_train = DCT(x_train, param["window_size"])  # (idx, ch, w, h)

    # plug trigger frequency
    for i in range(x_train.shape[0]):
        for ch in param["channel_list"]:
            for w in range(0, x_train.shape[2], param["window_size"]):
                for h in range(0, x_train.shape[3], param["window_size"]):
                    for pos in param["pos_list"]:
                        x_train[i][ch][w + pos[0]][h + pos[1]] += param["magnitude"]


    x_train = IDCT(x_train, param["window_size"])  # (idx, w, h, ch)

    if param["YUV"]:
        x_train = YUV2RGB(x_train)

    x_train /= 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train


def impose(x_train, y_train, param):
    x_train = poison_frequency(x_train, y_train, param)
    return x_train


def digest(param):
    txt = ""
    txt += param["dataset"]
    txt += str(param["target_label"])
    txt += str(param["poisoning_rate"])
    txt += str(param["label_dim"])
    txt += "".join(str(param["channel_list"]))
    txt += str(param["window_size"])
    txt += str(param["magnitude"])
    txt += str(param["YUV"])
    txt += "".join(str(param["pos_list"]))
    hash_md5 = md5()
    hash_md5.update(txt.encode("utf-8"))
    return hash_md5.hexdigest()
