import torch.optim as optim
from model import *
import util
from util import *
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, rho, pretrain, admm_training):
        self.model = gwnet(device, num_nodes, pretrain, admm_training, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.lrate = lrate
        self.wdecay = wdecay

        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = None

        #help variables
        self.O = None
        self.U = None
        self.rho = rho

        for name, param in self.model.named_parameters():
            if name == 'noise':
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lrate, weight_decay=wdecay)

    def train(self, input, real_val):
        self.model.train()
        input = nn.functional.pad(input, (1, 0, 0, 0))

        self.optimizer.zero_grad()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        #update loss with O and U
        loss = -self.loss(predict, real, 0.0) + (self.rho/2) * (torch.norm(self.model.noise - self.O + self.U))
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()


        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
