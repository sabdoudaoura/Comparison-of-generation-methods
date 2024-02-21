from networks import*
from utils import*
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)




def get_config(input_size,num_classes,layer_config,device):
    model = DNN(input_size,layer_config,num_classes,device)
    pretrain_model = DNN(input_size,layer_config,num_classes,device)
    return model,pretrain_model

def run_experiment(model,pretrain_model,pretraining:Args,training:Args):
    n_steps = pretraining.n_steps
    alpha = pretraining.alpha
    digits_test = pretraining.data
    batch_size = pretraining.batch_size
    k = pretraining.gibbs

    n_epochs = training.n_epochs
    lr = training.lr
    train_loader = training.train_loader
    test_loader = training.test_loader
    device = training.device
    model = model.to(device)
    digits_test = digits_test.to(device)
    
    pretrain_model = pretrain_model.to(device)
    print("[INFO] training model 1 from scratch")
    model.train(train_loader,n_epochs,lr)
    print("[INFO] Pretraining  model 2 ")
    pretrain_model.pretrain(digits_test,n_steps,alpha,batch_size,k)
    print("[INFO] training model 2")
    pretrain_model.train(train_loader,n_epochs,lr)

    print("[INFO] Evaluating models ")
    accuracy,pretrain_accuracy = model.evaluate(test_loader), pretrain_model.evaluate(test_loader)
    return accuracy,pretrain_accuracy