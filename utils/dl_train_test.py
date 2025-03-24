import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _default(batch, model):
    if len(batch) == 2:
        batchX, batchY = batch
        batchXs = [batchX]
    else:
        *batchXs, batchY = batch

    outputs = model(*batchXs)

    return outputs, batchY


def infer(model, device, testLoader, batchToInfer=_default):
    model.to(device)

    model.eval()

    predProbas = []
    actuals = []

    with torch.no_grad():
        for batch in testLoader:
            batch = [b.to(device) for b in batch]
            outputs, batchY = batchToInfer(batch, model)

            probas = torch.sigmoid(outputs)
            predProbas.append(probas)
            actuals.append(batchY)

    return torch.cat(predProbas), torch.cat(actuals)


def train(
    model,
    device,
    trainLoader,
    valLoader,
    criterion,
    optimizer,
    epochs=100,
    earlyStopping=10,
    batchToInfer=_default,
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

        for batch in trainLoader:
            batch = [b.to(device) for b in batch]
            outputs, batchY = batchToInfer(batch, model)
            loss = criterion(outputs, batchY)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * len(batchY)

        trainLoss /= len(trainLoader.dataset)
        trainLosses.append(trainLoss)

        # validation
        valPred, valActual = infer(model, device, valLoader, batchToInfer)
        valLoss = criterion(valPred, valActual)
        valLosses.append(valLoss.item())

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
