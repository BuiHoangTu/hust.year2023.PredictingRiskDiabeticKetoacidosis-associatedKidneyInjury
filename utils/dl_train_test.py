import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(
    model,
    device,
    trainLoader,
    valLoader,
    criterion,
    optimizer,
    epochs=100,
    earlyStopping=10,
):
    model.to(device)

    optimScheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)

    # early stopping
    bestValLoss = float("inf")
    bestModelState = None
    patience = 0

    trainLosses = []
    valLosses = []

    for epoch in range(epochs):
        model.train()
        trainLoss = 0

        for batchTsX, batchSX, batchY in trainLoader:
            batchTsX, batchSX = batchTsX.to(device), batchSX.to(device)
            batchY = batchY.to(device)

            outputs = model(batchTsX, batchSX)
            loss = criterion(outputs, batchY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * batchTsX.size(0)

        trainLoss /= len(trainLoader.dataset)
        trainLosses.append(trainLoss)

        # validation
        model.eval()
        valLoss = 0

        with torch.no_grad():
            for batchTsX, batchSX, batchY in valLoader:
                batchTsX, batchSX = batchTsX.to(device), batchSX.to(device)
                batchY = batchY.to(device)

                outputs = model(batchTsX, batchSX)
                loss = criterion(outputs, batchY)

                valLoss += loss.item() * batchTsX.size(0)

        valLoss /= len(valLoader.dataset)
        valLosses.append(valLoss)

        optimScheduler.step(valLoss)

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            bestModelState = model.state_dict()
            patience = 0
        else:
            patience += 1

        print(
            f"Epoch {epoch+1}/{epochs} Train Loss: {trainLoss:.4f} Val Loss: {valLoss:.4f}"
        )

        if patience > earlyStopping:
            break

    model.load_state_dict(bestModelState)
    return model, trainLosses, valLosses


def predTest(model, device, testLoader):
    model.eval()

    predProbas = []
    actuals = []

    with torch.no_grad():
        for batchTsX, batchSX, batchY in testLoader:
            batchTsX, batchSX = batchTsX.to(device), batchSX.to(device)
            batchY = batchY.to(device)

            outputs = model(batchTsX, batchSX)
            probas = torch.sigmoid(outputs)
            predProbas.extend(probas.cpu().numpy())
            actuals.extend(batchY.cpu().numpy())

    return predProbas, actuals
