from torch import nn
from metric import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



def validate(validation_loader, device, model, criterion):
    model = model.to(device)
    model = model.eval() # set eval mode
    with torch.no_grad(): # network does not update gradient during evaluation
        val_top1 = AverageMeter()
        validate_loader = tqdm(validation_loader)
        validate_loss = 0
        for i, data in enumerate(validate_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            prec1, prec2 = accuracy(outputs, labels, topk=(1,2))
            n = inputs.size(0) # barch_size
            val_top1.update(prec.item(), n)
            validate_loss += loss.item()
            postfix = {'validation_loss':%.6f%(validate_loss/(i+1)), 'validation_acc':'%.6f'%val_top1.avg}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return val_acc

def submission(test_loader, device, model):
    result_list = []
    model = model.to(device)
    test_loader = tqdm(test_loader)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)

            softmax_func = nn.Softmax(dim=1) # dim=1 means the sum of rows is 1
            soft_output = softmax_func(outputs) # soft_output is become two probability value
            predicted = soft_output[:, 1] # the probability of dog

            for i in range(len(predicted)):
                result_list.append({'id':labels[i].item(), 'label':predicted[i].item()})
    return result_list









    




    
