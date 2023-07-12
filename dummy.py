
print("Training starts now")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = GradScaler()
for epoch in range(num_epochs):
    train_running_loss = 0.0
    train_acc = 0.0

    model = model.train()

    ## training step
    for i, (images, labels) in enumerate(trainloader):

        images = images.to(device)
        labels = labels.to(device)

        ## forward + backprop + loss
        with autocast(device_type='cuda', dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        ## update model params
        scaler.step(optimizer)
        scaler.update()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(logits, labels, BATCH_SIZE)

    model.eval()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
            %(epoch, train_running_loss / i, train_acc/i))
