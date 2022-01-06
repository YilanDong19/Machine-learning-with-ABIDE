from torch.autograd import Variable
from save_load_model import *



def enable_dropout(model):
  for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()
def Hp(x):
    if x == 1:
        x = x - 1e-6
    elif x < 1e-6:
        x = x + 1e-6
    else:
        x = x
    return - (x * torch.log2(x)) - ((1 - x) * torch.log2(1 - x))

def train_GCN(args,epoch,  model, all_data, fold_train_index, fold_validation_index, fold_test_index, scheduler, fold, k_fold, save_path, Sheet1, position):

    model.train()
    whole_number = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    data = torch.from_numpy(all_data[0])
    data = Variable(data, requires_grad=True)
    data = data.float()

    target = torch.from_numpy(all_data[1])
    target = target.float().long()

    edge_index = torch.from_numpy(all_data[2])
    edge_index = Variable(edge_index, requires_grad=False)

    edgenet_input = torch.from_numpy(all_data[3])
    edgenet_input = Variable(edgenet_input, requires_grad=True)
    edgenet_input = edgenet_input.float()
    if args.cuda:
        data, target, edge_index, edgenet_input = data.cuda(), target.cuda(), edge_index.cuda(), edgenet_input.cuda()

    scheduler.zero_grad()
    out = model(data, edge_index, edgenet_input)
    out = out[fold_train_index,:]
    target = target[fold_train_index, :]
    cross_loss = torch.nn.functional.nll_loss(out, torch.max(target, 1)[1])
    out = torch.max(out, 1)[1]
    target = torch.max(target, 1)[1]
    target = target.cpu().numpy()
    out = out.cpu().numpy()
    cross_loss.backward()

    for i in range(len(target)):
        if target[i] == 1 and out[i] == 1:
            TN = TN + 1
            whole_number = whole_number + 1
        elif target[i] == 1 and out[i] == 0:
            FP = FP + 1
            whole_number = whole_number + 1
        elif target[i] == 0 and out[i] == 1:
            FN = FN + 1
            whole_number = whole_number + 1
        elif target[i] == 0 and out[i] == 0:
            TP = TP + 1
            whole_number = whole_number + 1

    accuracy = (TP + TN) / whole_number
    scheduler.step()


    print(
                'Train Epoch: ' + str(epoch) + '_' + str(
                fold) + ': train loss : {:.8f}\t train accuracy : {:.8f}\t'.format(
                    cross_loss.item(), accuracy))
    save_model(model, save_path, str(epoch) + '_' + str(fold))
    Sheet1.cell(row=fold+position[0], column=epoch+1, value=accuracy)
    Sheet1.cell(row=fold+position[1], column=epoch+1, value=cross_loss.item())
    ############################## validation and test ################################################################################
    test_GCN(args, epoch, model, all_data, fold_test_index, Sheet1, position, fold)
    validation_GCN(args, epoch, model, all_data, fold_validation_index, fold, Sheet1, position)


def validation_GCN(args, epoch, model,all_data,  fold_validation_index, fold, Sheet1, position):

    model.eval()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    whole_number = 0

    data = torch.from_numpy(all_data[0])
    data = Variable(data, requires_grad=False)
    data = data.float()

    target = torch.from_numpy(all_data[1])
    target = target.float().long()

    edge_index = torch.from_numpy(all_data[2])
    edge_index = Variable(edge_index, requires_grad=False)

    edgenet_input = torch.from_numpy(all_data[3])
    edgenet_input = Variable(edgenet_input, requires_grad=False)
    edgenet_input = edgenet_input.float()

    if args.cuda:
        data, target, edge_index, edgenet_input = data.cuda(), target.cuda(), edge_index.cuda(), edgenet_input.cuda()


    out = model(data, edge_index, edgenet_input)
    out = out[fold_validation_index, :]
    target = target[fold_validation_index, :]
    cross_loss = torch.nn.functional.nll_loss(out, torch.max(target, 1)[1])

    out = torch.max(out, 1)[1]
    target = torch.max(target, 1)[1]
    target = target.cpu().numpy()
    out = out.cpu().numpy()

    for i in range(len(target)):
        if target[i] == 1 and out[i] == 1:
            TN = TN + 1
            whole_number = whole_number + 1
        elif target[i] == 1 and out[i] == 0:
            FP = FP + 1
            whole_number = whole_number + 1
        elif target[i] == 0 and out[i] == 1:
            FN = FN + 1
            whole_number = whole_number + 1
        elif target[i] == 0 and out[i] == 0:
            TP = TP + 1
            whole_number = whole_number + 1

    accuracy = (TP + TN) / whole_number
    if TP + FP != 0:
        precision = TP / (TP + FP)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        print(
            'model: ' + str(epoch) + '_' + str(
                fold) + ': accuracy : {:.8f}\t average_loss : {:.8f}\t precision : {:.8f}\t sensitivity : {:.8f}\t specificity : {:.8f}\t'.format(
                accuracy, cross_loss.item(), precision, sensitivity, specificity))
    else:
        print(
            'Validation: ' + str(epoch) + '_' + str(fold) + ': accuracy : {:.8f}\t average_loss : {:.8f}\t'.format(
                accuracy, cross_loss.item()))
    Sheet1.cell(row=fold + position[0] + position[2], column=epoch + 1, value=accuracy)
    Sheet1.cell(row=fold + position[1] + position[2], column=epoch + 1, value=cross_loss.item())



def test_GCN(args, epoch, model, all_data,  fold_test_index, Sheet1, position, fold):
    model.eval()
    whole_number = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    data = torch.from_numpy(all_data[0])
    data = Variable(data, requires_grad=False)
    data = data.float()

    target = torch.from_numpy(all_data[1])
    target = target.float().long()

    edge_index = torch.from_numpy(all_data[2])
    edge_index = Variable(edge_index, requires_grad=False)

    edgenet_input = torch.from_numpy(all_data[3])
    edgenet_input = Variable(edgenet_input, requires_grad=False)
    edgenet_input = edgenet_input.float()

    if args.cuda:
        data, target, edge_index, edgenet_input = data.cuda(), target.cuda(), edge_index.cuda(), edgenet_input.cuda()

    out = model(data, edge_index, edgenet_input)
    out = out[fold_test_index, :]
    target = target[fold_test_index, :]
    cross_loss = torch.nn.functional.nll_loss(out, torch.max(target, 1)[1])

    out = torch.max(out, 1)[1]
    target = torch.max(target, 1)[1]
    target = target.cpu().numpy()
    out = out.cpu().numpy()

    for i in range(len(target)):
        if target[i] == 1 and out[i] == 1:
            TN = TN + 1
            whole_number = whole_number + 1
        elif target[i] == 1 and out[i] == 0:
            FP = FP + 1
            whole_number = whole_number + 1
        elif target[i] == 0 and out[i] == 1:
            FN = FN + 1
            whole_number = whole_number + 1
        elif target[i] == 0 and out[i] == 0:
            TP = TP + 1
            whole_number = whole_number + 1

    accuracy = (TP + TN) / whole_number
    Sheet1.cell(row=fold + position[0] + position[2] * 2, column=epoch + 1, value=accuracy)
    Sheet1.cell(row=fold + position[1] + position[2] * 2, column=epoch + 1, value=cross_loss.item())

    if TP + FP != 0:
        precision = TP / (TP + FP)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        print(
            'model: ' + str(epoch) + '_' + str(
                fold) + ': accuracy : {:.8f}\t average_loss : {:.8f}\t precision : {:.8f}\t sensitivity : {:.8f}\t specificity : {:.8f}\t'.format(
                accuracy, cross_loss.item(), precision, sensitivity, specificity))
        Sheet1.cell(row=fold + position[0] + position[2] * 3, column=epoch + 1, value=sensitivity)
        Sheet1.cell(row=fold + position[1] + position[2] * 3, column=epoch + 1, value=specificity)
    else:
        print(
            'model: ' + str(epoch) + '_' + str(fold) + ': accuracy : {:.8f}\t average_loss : {:.8f}\t'.format(
                accuracy, cross_loss.item()))
        Sheet1.cell(row=fold + position[0] + position[2] * 3, column=epoch + 1, value=0)
        Sheet1.cell(row=fold + position[1] + position[2] * 3, column=epoch + 1, value=0)

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch == 1:
            lr = 4e-6
        elif epoch == 50:
            lr = 4e-6
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if optAlg == 'adam':
        if epoch == 1:
            lr = 8e-4
        elif epoch == 100:
            lr = 8e-4
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
