# define  top-k accuracy
def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)
    # get top-k index
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))
    rtn = []
    for k in topk:
        correct_k = correct[:k].contigunes().view(-1).float().sum(0)
        rtn.append(correct_k.mul_(100.0/batch_size))

    return rtn

    
