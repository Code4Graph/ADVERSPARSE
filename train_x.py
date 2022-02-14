import torch
import numpy as np
import argparse
import time
import util
from util import *
import matplotlib.pyplot as plt
import copy
from engine_x import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer', default=True)
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true',help='whether add adaptive adj', default=True)
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj', default=True)
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=5,help='') 
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--keep',type=float,default=0.5,help='keeping rate')
parser.add_argument('--ADMMtimes',type=int,default=5,help='keeping rate')
parser.add_argument('--rho',type=float,default=0.5,help='keeping rate')
parser.add_argument('--threshold',type=float,default=0.5,help='keeping rate')
parser.add_argument('--pretrain',type=bool,default=False,help='pretrain')
parser.add_argument('--admm_training',type=bool,default=True,help='admm training')
parser.add_argument('--best_pretrained_model',type=str,default='_exp1_best_2.95.pth',help='best pretrained model')
args = parser.parse_args()


print("----------------------------------")
print("admm_training: ", args.admm_training)
print("pretrain: ", args.pretrain)
print("best pretrained model: ", args.best_pretrained_model)
print("keep rate: ", args.keep)
print("rho: ", args.rho)
print("ADMM TIMES: ", args.ADMMtimes)
print("epoch: ", args.epochs)
print("threshold: ", args.threshold)
print("----------------------------------")

def main():
    rho = args.rho
    keep = args.keep
    ADMM_times = args.ADMMtimes
    best_model_path = args.best_pretrained_model
    threshold = args.threshold

    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
    ori_adj = supports[0].cpu().data.numpy()


    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, rho, args.pretrain, args.admm_training)


    # #########################################################################
    # #load the pre-trained model and test
    # #########################################################################
    print("load model.....",flush=True)
    engine.model.load_state_dict(torch.load(args.save + best_model_path))
    print("test load model.......")
    engine.model.eval()
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


    ##########################################################################################
    #start train the x
    ##########################################################################################
    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    adj = supports[0].cpu().data.numpy()

    engine.O = torch.tensor(np.ones_like(adj)).to(device)
    engine.U = torch.tensor(initialize(adj)).to(device)
    new_noise = engine.model.noise.cpu().data.numpy()
    filter= np.eye(new_noise.shape[0])
    filter[filter == 1] = 2
    filter[filter == 0] = 1
    filter[filter == 2] = 0

    # init ones
    updated_new_noise = np.ones_like(adj)
    updated_new_noise_torch = torch.tensor(np.multiply(updated_new_noise, filter), dtype=torch.float32).to(device)
    engine.model.noise.data = updated_new_noise_torch
    count = 0
    for j in range(ADMM_times):
        print("current admm times: ", j)
        for i in range(1,args.epochs+1):
            count += 1
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = torch.Tensor(x).to(device)
                trainx = trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)
                metrics = engine.train(trainx, trainy[:, 0, :, :])
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])
                if iter % args.print_every == 0:
                    log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                    print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
            t2 = time.time()
            train_time.append(t2 - t1)

            ################################################################
            # Validation
            ################################################################
            valid_loss = []
            valid_mape = []
            valid_rmse = []
            s1 = time.time()
            for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:, 0, :, :])
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            s2 = time.time()
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            print(log.format(i, (s2 - s1)))
            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
            torch.save(engine.model.state_dict(), args.save + "_admm_epoch_" + str(count) + "_" + str(round(mvalid_loss, 2)) + ".pth")


        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        print("update admm O and U")
        engine.O = engine.model.noise + engine.U
        engine.O = torch.tensor(cutRoad(engine.O.cpu().data.numpy(), ori_adj, keep_ratio=keep, margin=threshold), dtype=torch.float32).to(device)
        engine.U = engine.U + engine.model.noise - engine.O

    #coverge and project
    print("finish admm training")
    new_noise = torch.tensor(cutRoad(engine.model.noise.cpu().data.numpy(), ori_adj, keep_ratio=keep, margin=threshold), dtype=torch.float32).to(device)
    engine.model.noise.data = new_noise
    torch.save(engine.model.state_dict(), args.save + "_converge_.pth")



    ###########################################################################
    #testing
    ###########################################################################
    engine.model.load_state_dict(torch.load(args.save+ "_converge_.pth"))
    engine.model.eval()
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1,3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]
    print("Training finished")


    amae = []
    amape = []
    armse = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}' #attention
    print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
