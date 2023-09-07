import numpy as np
import torch
import torch.nn as nn
from customdataset import loaders
import tqdm
from torch.optim import lr_scheduler, optim
from validation import validation
from torch.autograd import Variable

# 하이퍼파라미터 그리드 설정
param_grid = {
    'lr': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128],
    'hidden_size': [8, 64, 128],
    'optimizer': ['Adam', 'RMSprop', 'SGD']
}

lr = param_grid['lr'][0]
batch_size = param_grid['batch_size'][0]
num_lstm_out = param_grid['hidden_size'][0]
optimizer_name = param_grid['optimizer'][0]

randomseedlist=[4444,2514,4040,8282,1004]



def final(modelname,train_set, train_label, test_set, test_label,esbnum):
    train_loader,valid_loader,test_loader=loaders(train_set, train_label,test_set,test_label,randomseedlist[esbnum])


    valid_loss_min = np.Inf # track change in validation loss
    criterion = nn.NLLLoss().to(device)
    epochs=500
    print_every=1000

    # 0.001 32 8 Adam
    lr = param_grid['lr'][0]
    batch_size = param_grid['batch_size'][0]
    num_lstm_out = param_grid['hidden_size'][0]
    optimizer_name = param_grid['optimizer'][0]

    model =modelname.to(device)


    # 옵티마이저 설정
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)



    #train
    steps = 0

    for e in tqdm(range(epochs)):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for inputs, labels in train_loader:
            steps += 1

            inputs = inputs.float()
            inputs, labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()
            # print(inputs.shape)
            # print(model.forward(inputs).shape)
            output,_ = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.6f}.. ".format(train_loss/print_every),
                      "Val Loss: {:.6f}.. ".format(valid_loss/len(valid_loader)),
                      "Val Accuracy: {:.2f}%".format(accuracy/len(valid_loader)*100))

                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                    torch.save(model.state_dict(), savepath+ 'malstm_msn_cv_'+str(cv)+'_cond_'+str(cond)+'_esb_'+str(esbnum)+'.pt')
                    valid_loss_min = valid_loss

                train_loss = 0

                model.train()
    return model