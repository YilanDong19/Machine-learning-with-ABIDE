from torch.autograd import Variable
from save_load_model import *
from DNN import *


def train_DNN(args,epoch,  model, trainLoader, validationLoader, testLoader, scheduler, fold, save_path, Sheet1, position):

    model.train()
    nProcessed = 0
    average_accuracy = 0
    average_loss = 0
    number_batch = 0
    nTrain = len(trainLoader.dataset)
    whole_number = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for batch_idx, (train, label, index) in enumerate(trainLoader):
        data = train
        target = label
        number_batch = number_batch + 1

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data= Variable(data, requires_grad=True)
        scheduler.zero_grad()
        data = data.float()
        out = model(data)
        target = target.float().long()
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
        average_accuracy = average_accuracy + accuracy
        average_loss = average_loss + cross_loss

        scheduler.step()
        nProcessed += len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1

        print(
                'Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\t train loss : {:.8f}\t train accuracy : {:.8f}\t'.format(
                    partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
                    cross_loss.item(), accuracy))

    save_model(model, save_path, str(epoch) + '_' + str(fold))
    average_accuracy = average_accuracy / number_batch
    average_loss = average_loss / number_batch

    Sheet1.cell(row=fold+position[0], column=epoch+1, value=average_accuracy)
    Sheet1.cell(row=fold+position[1], column=epoch+1, value=average_loss.item())
        ############################## validation ################################################################################
    test_DNN(args, epoch, model, testLoader, Sheet1, position, fold)
    validation_DNN(args, epoch, model, validationLoader, fold, Sheet1, position)




def validation_DNN(args, epoch, model, validationLoader, fold, Sheet1, position):

    model.eval()
    val_number_batch = 0
    val_average_loss = 0
    val_whole_number = 0
    val_TP = 0
    val_FP = 0
    val_TN = 0
    val_FN = 0

    for val_batch_idx, (val_train, val_label, val_index) in enumerate(validationLoader):
        val_data = val_train
        val_target = val_label
        if args.cuda:
            val_data, val_target = val_data.cuda(), val_target.cuda()
        val_data = val_data.float()
        val_number_batch = val_number_batch + 1

        val_out = model(val_data)
        val_cross_loss = torch.nn.functional.nll_loss(val_out, torch.max(val_target, 1)[1])
        val_out = torch.max(val_out, 1)[1]
        val_target = torch.max(val_target, 1)[1]
        val_target = val_target.cpu().numpy()
        val_out = val_out.cpu().numpy()
        val_average_loss = val_cross_loss + val_average_loss
        for i in range(len(val_target)):
            if val_target[i] == 1 and val_out[i] == 1:
                val_TN = val_TN + 1
                val_whole_number = val_whole_number + 1
            elif val_target[i] == 1 and val_out[i] == 0:
                val_FP = val_FP + 1
                val_whole_number = val_whole_number + 1
            elif val_target[i] == 0 and val_out[i] == 1:
                val_FN = val_FN + 1
                val_whole_number = val_whole_number + 1
            elif val_target[i] == 0 and val_out[i] == 0:
                val_TP = val_TP + 1
                val_whole_number = val_whole_number + 1

    val_accuracy = (val_TP + val_TN) / val_whole_number
    val_average_loss = val_average_loss / val_number_batch


    if val_TP + val_FP != 0:
        precision = val_TP / (val_TP + val_FP)
        sensitivity = val_TP / (val_TP + val_FN)
        specificity = val_TN / (val_TN + val_FP)
        print(
            'model: ' + str(epoch) + '_' + str(
                fold) + ': accuracy : {:.8f}\t average_loss : {:.8f}\t precision : {:.8f}\t sensitivity : {:.8f}\t specificity : {:.8f}\t'.format(
                val_accuracy, val_average_loss, precision, sensitivity, specificity))
    else:
        print(
            'Validation: ' + str(epoch) + '_' + str(fold) + ': accuracy : {:.8f}\t average_loss : {:.8f}\t'.format(
                val_accuracy, val_average_loss))
    Sheet1.cell(row=fold + position[0] + position[2], column=epoch + 1, value=val_accuracy)
    Sheet1.cell(row=fold + position[1] + position[2], column=epoch + 1, value=val_average_loss.item())



def test_DNN(args, epoch, model, testLoader, Sheet1, position, fold):

    model.eval()
    number_batch = 0
    average_loss = 0
    whole_number = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0


    for batch_idx, (train, label, index) in enumerate(testLoader):
        data = train
        target = label
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.float()
        number_batch= number_batch+1

        out = model(data)
        cross_loss = torch.nn.functional.nll_loss(out, torch.max(target, 1)[1])
        out = torch.max(out, 1)[1]
        target = torch.max(target, 1)[1]
        target = target.cpu().numpy()
        out = out.cpu().numpy()
        average_loss = cross_loss + average_loss
        for i in range(len(target)):
            if target[i] ==1 and out[i] == 1:
                TN = TN + 1
                whole_number = whole_number+1
            elif target[i] ==1 and out[i] == 0:
                FP = FP + 1
                whole_number = whole_number + 1
            elif target[i] ==0 and out[i] == 1:
                FN = FN + 1
                whole_number = whole_number + 1
            elif target[i] ==0 and out[i] == 0:
                TP = TP + 1
                whole_number = whole_number + 1

    accuracy = (TP + TN)/whole_number
    average_loss = average_loss/number_batch
    Sheet1.cell(row=fold + position[0] + position[2] * 2, column=epoch + 1, value=accuracy)
    Sheet1.cell(row=fold + position[1] + position[2] * 2, column=epoch + 1, value=average_loss.item())

    if TP + FP != 0:
        precision = TP / (TP + FP)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        print(
            'model: '+ str(epoch)+'_'+str(fold)+': accuracy : {:.8f}\t average_loss : {:.8f}\t precision : {:.8f}\t sensitivity : {:.8f}\t specificity : {:.8f}\t'.format(
                accuracy, average_loss, precision, sensitivity, specificity))
        Sheet1.cell(row=fold + position[0] + position[2] * 3, column=epoch+1, value=sensitivity)
        Sheet1.cell(row=fold + position[1] + position[2] * 3, column=epoch+1, value=specificity)
    else:
        print(
            'model: '+ str(epoch)+'_'+str(fold)+': accuracy : {:.8f}\t average_loss : {:.8f}\t'.format(
                accuracy, average_loss))
        Sheet1.cell(row=fold + position[0] + position[2] * 3, column=epoch + 1, value=0)
        Sheet1.cell(row=fold + position[1] + position[2] * 3, column=epoch + 1, value=0)

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch == 1:
            lr = 5e-6
        elif epoch == 50:
            lr = 5e-7
        else:
            return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if optAlg == 'adam':
        if epoch == 1:
            lr = 2e-5
        elif epoch == 100:
            lr = 2e-5

        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_opt_auto(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch == 1:
            lr = 5e-6
        elif epoch == 50:
            lr = 5e-7

        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if optAlg == 'adam':
        if epoch == 1:
            lr = 2e-4
        elif epoch == 100:
            lr = 2e-4
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
