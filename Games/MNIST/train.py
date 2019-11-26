import torch.utils.data
from torch import nn
from torch.autograd import Variable
from Dataset import Minst_Dataset
from model import CharNet
import torch.optim.lr_scheduler
import Models.genetic as genetic

BATCH_SIZE = 512
NUM_EPOCHS = 30
LR_DECAY = 10


def main():
    # Load model, dataset and set up gradient decent
    pop_size = 50
    models = [CharNet(10) for _ in range(pop_size)]
    for model in models:
        model.cuda()

    dataset = Minst_Dataset('train.csv')

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )



    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        losses = []
        for model in models:
            curr_loss = train(model, train_loader, criterion)
            losses.append(curr_loss)
        selections = genetic.select_agents(losses, 0.2)
        parents = [models[selections[i][0]] for i in range(len(selections))]
        new_models = genetic.crossover(parents, pop_size, CharNet)
        genetic.mutate_agents(new_models, 0.1, 0.01)
        models = new_models
        for model in models:
            model.cuda()


    # Save the model
    torch.save(model.state_dict(), "wieghts.pth.tar")



def train(model, train_loader, criterion):
    model.train()
    total_loss = 0
    for batch_num, (data, label) in enumerate(train_loader):
        data = data.cuda()
        label = label.cuda()

        output = model(data)

        loss = criterion(output, label)
        total_loss += loss.item()


    print(total_loss)
    return total_loss





if __name__ == "__main__":
    main()