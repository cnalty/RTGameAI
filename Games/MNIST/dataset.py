import pandas as pd
import torch
import numpy as np

class Mnist_Dataset():
    def __init__(self, data_file, type="train"):
        self.type = type
        raw_data = pd.read_csv(data_file)
        self.data = np.array(raw_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.type == "train":
            data = self.data[index][1:]/255
            label = self.data[index][0]
        else:
            data = self.data / 255
            label = np.nan
        data = data.reshape(28,-1)
        data = [data]
        data = torch.tensor(data).float()



        return data, label


def main():
    from PIL import Image

    w, h = 28, 28
    dataset = Mnist_Dataset('train.csv')
    for item, label in dataset:
        item = item.numpy()
        item = (item * 255).astype(np.uint8)[0]
        print(item)
        img = Image.fromarray(item)
        img.show()


if __name__ == "__main__":
    main()