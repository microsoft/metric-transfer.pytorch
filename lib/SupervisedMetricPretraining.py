import torch
from torch import nn
train_labels_= None
def get_train_labels(trainloader, device='cuda'):
    global train_labels_
    if train_labels_ is None:
        print("=> loading all train labels")
        train_labels = -1 * torch.ones([len(trainloader.dataset)], dtype=torch.long)
        for i, (_, label, index) in enumerate(trainloader):
            train_labels[index] = label
            if i % 10000 == 0:
                print("{}/{}".format(i, len(trainloader)))
        assert all(train_labels != -1)
        train_labels_ = train_labels.to(device)
    return train_labels_
class Supervised_Pretraining(object):
    def __init__(self,trainloader,n, t=0.07):
        """
        Parameters
        ----------
        trainloader : 
            DataLoader containing training data.
        n : int
            Number of labels.
        t : float
            Temperature parameter as described in https://arxiv.org/pdf/1805.01978.pdf.

        """
        super(Supervised_Pretraining,self).__init__()
        # get train labels
        self.labels = get_train_labels(trainloader)
        # Softmax loss
        self.loss_fn = nn.CrossEntropyLoss()
        #init labels
        self.n_labels = n
        self.t = t
    def to(self,device):
        #send to a device
        self.loss_fn.to(device)
    def __call__(self,out,y):
        return self.forward(out,y)
    def forward(self,out,y):
        """
        Parameters
        ----------
        out : 
            Output from LinearAverage.py as described in https://arxiv.org/pdf/1812.08781.pdf.
        y : tensor
            Target Labels.

       
          Returns
        -------
        Softmax Loss.

        """
        #making it more sensitive by dividing by temperature value as in https://arxiv.org/pdf/1805.01978.pdf
        out.div_(self.t)
        #eq (4) in https://arxiv.org/pdf/1812.08781.pdf
        scores = torch.zeros(out.shape[0],self.n_labels).cuda()
        for i in range(self.n_labels):
            yi = self.labels == i

            candidates = yi.view(1,-1).expand(out.shape[0], -1)
            retrieval = out[candidates]
            retrieval = retrieval.reshape(out.shape[0], -1)

            scores[:,i] = retrieval.sum(1,keepdim=True).view(1,-1)

        return self.loss_fn(scores, y)