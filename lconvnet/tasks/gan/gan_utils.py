import os, gzip, torch, argparse, json
from munch import Munch
import random
import ast
import shlex
import collections
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets


def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + "/train-images-idx3-ubyte.gz", 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + "/train-labels-idx1-ubyte.gz", 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + "/t10k-images-idx3-ubyte.gz", 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + "/t10k-labels-idx1-ubyte.gz", 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.0
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))

    plt.figure(0)
    if images.shape[3] in (3, 4):
        plt.imshow(image)
    elif images.shape[3] == 1:
        plt.imshow(image, cmap='Greys_r')
    else:
        raise ValueError(
            "in merge(images,size) images parameter "
            "must have dimensions: HxW or HxWx3 or HxWx4"
        )
    plt.savefig(path)
    plt.close("all")
    return None


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h : j * h + h, i * w : i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h : j * h + h, i * w : i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError(
            "in merge(images,size) images parameter "
            "must have dimensions: HxW or HxWx3 or HxWx4"
        )


def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + "_epoch%03d" % (e + 1) + ".png"
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + "_generate_animation.gif", images, fps=5)


def loss_plot(hist, path="Train_hist.png", model_name=""):
    x = range(len(hist["D_loss"]))

    y1 = hist["D_loss"]
    y2 = hist["G_loss"]

    plt.plot(x, y1, label="D_loss")
    plt.plot(x, y2, label="G_loss")

    plt.xlabel("Iter")
    plt.ylabel("Loss")

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + "_loss.png")

    plt.savefig(path)

    plt.close()


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def process_config(verbose=True):
    args = get_config_overrides()
    config = json.load(open(args.config))
    if args.o is not None:
        print(args.o)
        config = update(config, args.o)

    if verbose:
        import pprint

        pp = pprint.PrettyPrinter()
        print("-------- Config --------")
        pp.pprint(config)
        print("------------------------")

    # Use a munch object for ease of access. Munch is almost the same as Bunch, but better integrated with Python 3.
    config = Munch.fromDict(config)

    return config


def get_config_overrides():
    parser = argparse.ArgumentParser(description="Experiments with Lipschitz networks")
    parser.add_argument("config", help="Base config file")
    parser.add_argument(
        "-o",
        action=ConfigParse,
        help="Config option overrides. Separated like: e.g. optim.lr_init=1.0,,optim.lr_decay=0.1",
    )
    return parser.parse_args()


def set_experiment_seed(seed):
    # Set the seed.
    np.random.seed(seed)

    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            if "+" in v:
                # This typing is a bit hacky
                # Assumes something is in the list
                v = [type(d[k][0])(x) for x in v.split("+")]
            try:
                d[k] = type(d[k])(v)
            except (TypeError, ValueError) as e:
                raise TypeError(e)  # Types not compatible.
            except KeyError:
                d[k] = v  # No matching key in dict.
    return d


class ConfigParse(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        options_dict = {}
        for overrides in shlex.split(values):
            k, v = overrides.split("=")
            k_parts = k.split(".")
            dic = options_dict
            for key in k_parts[:-1]:
                dic = dic.setdefault(key, {})
            if v.startswith("[") and v.endswith("]"):
                v = ast.literal_eval(v)
            dic[k_parts[-1]] = v
        setattr(namespace, self.dest, options_dict)
