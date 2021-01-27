import torch
from torch import nn
from .utils import get_train_labels

class SupervisedSoftmax(object):
    def __init__(self,trainloader,device,t=0.07):
        super(SupervisedSoftmax,self).__init__()
        # get train labels
        self.labels = get_train_labels(trainloader)
        # Softmax loss
        self.loss_fn = nn.CrossEntropyLoss().to(device)
        #init labels
        self.n_labels = self.labels.max().data.item() + 1
        print(self.n_labels)
        #Temperature parameter as described in https://arxiv.org/pdf/1805.01978.pdf.
        self.temperature = t
    def to(self,device):
        #send to a device
        self.loss_fn.to(device)
    def __call__(self,dist,y):
        return self.forward(dist,y)
    def forward(self,dist,y):
        #making it more sensitive by dividing by temperature value as in https://arxiv.org/pdf/1805.01978.pdf
        dist.div_(self.temperature)
        #eq (4) in https://arxiv.org/pdf/1812.08781.pdf
        scores = torch.zeros(dist.shape[0],self.n_labels).cuda()
        for i in range(self.n_labels):
            yi = self.labels == i
            candidates = yi.view(1,-1).expand(dist.shape[0], -1)
            retrieval = dist[candidates]
            retrieval = retrieval.reshape(dist.shape[0], -1)
            scores[:,i] = retrieval.sum(1,keepdim=True).view(1,-1)
        
        return self.loss_fn(scores, y)
