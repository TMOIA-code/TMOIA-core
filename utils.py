import math
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from model import *

torch.set_default_dtype(torch.float16)

class dataset(Dateset.Dataset):
    def __init__(self, snp, mRNA, mCG, mCHG, mCHH, labels):
        self.snp = snp.to_numpy()
        self.mRNA = mRNA.to_numpy()
        self.mCG = mCG.to_numpy()
        self.mCHG = mCHG.to_numpy()
        self.mCHH = mCHH.to_numpy()
        self.labels = labels.to_numpy()

    def __len__(self):
        return len(self.snp)

    def __getitem__(self, index):
        data_snp = torch.Tensor(self.snp[index])
        data_mRNA = torch.Tensor(self.mRNA[index])
        data_mCG = torch.Tensor(self.mCG[index])
        data_mCHG = torch.Tensor(self.mCHG[index])
        data_mCHH = torch.Tensor(self.mCHH[index])
        labels = torch.tensor(self.labels[index])

        return data_snp, data_mRNA, data_mCG, data_mCHG, data_mCHH, labels

class dataset_feature_rank(Dateset.Dataset):
    def __init__(self, snp, mRNA, mCG, mCHG, mCHH, labels):
        self.snp = snp
        self.mRNA = mRNA
        self.mCG = mCG
        self.mCHG = mCHG
        self.mCHH = mCHH
        self.labels = labels

    def __len__(self):
        return len(self.snp)

    def __getitem__(self, index):
        data_snp = torch.Tensor(self.snp[index])
        data_mRNA = torch.Tensor(self.mRNA[index])
        data_mCG = torch.Tensor(self.mCG[index])
        data_mCHG = torch.Tensor(self.mCHG[index])
        data_mCHH = torch.Tensor(self.mCHH[index])
        labels = torch.tensor(self.labels[index])

        return data_snp, data_mRNA, data_mCG, data_mCHG, data_mCHH, labels

def input_data(snp, mRNA, mCG, mCHG, mCHH, labels, feature_num):
    mRNA = pd.read_csv(mRNA, index_col=0)
    mCG = pd.read_csv(mCG, index_col=0)
    mCHG = pd.read_csv(mCHG, index_col=0)
    mCHH = pd.read_csv(mCHH, index_col=0)
    snp = pd.read_csv(snp, index_col=0)
    idx = mRNA.index.to_list()

    mRNA = mRNA.to_numpy().astype(float)
    mCG = mCG.to_numpy().astype(float)
    mCHG = mCHG.to_numpy().astype(float)
    mCHH = mCHH.to_numpy().astype(float)
    snp = snp.to_numpy().astype(float)

    labels = pd.read_table(labels, header=None, index_col=0)
    # labels = pd.read_csv(labels, index_col=0)
    labels_np = labels.to_numpy().astype(float)
    labels_np = labels_np.reshape(-1,)

    mRNA = SelectKBest(f_regression, k=feature_num).fit_transform(mRNA, labels_np)
    mCG = SelectKBest(f_regression, k=feature_num).fit_transform(mCG, labels_np)
    mCHH = SelectKBest(f_regression, k=feature_num).fit_transform(mCHH, labels_np)
    mCHG = SelectKBest(f_regression, k=feature_num).fit_transform(mCHG, labels_np)
    snp = SelectKBest(f_regression, k=feature_num).fit_transform(snp, labels_np)

    mRNA = pd.DataFrame(mRNA).set_index([idx])
    mCG = pd.DataFrame(mCG).set_index([idx])
    mCHG = pd.DataFrame(mCHG).set_index([idx])
    mCHH = pd.DataFrame(mCHH).set_index([idx])
    snp = pd.DataFrame(snp).set_index([idx])

    return snp, mRNA, mCG, mCHG, mCHH, labels

def computeCorrelation(X,Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0,len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = math.sqrt(varX * varY)
    return SSR / SST

def MAE(X, Y):
    mae = 0
    for i in range(0, len(X)):
        x = X[i]
        y = Y[i]
        ABS = abs(x-y)
        mae += ABS
    Mae = mae / len(X)
    return Mae

def MSE(X, Y):
    mse = 0
    for i in range(0, len(X)):
        x = X[i]
        y = Y[i]
        square = (x-y) ** 2
        mse += square
    Mse = mse / len(X)
    return Mse

def train_val(config, epochs, learning_rate, dataloader_tr, dataloader_val, dataloader_te):
    # save_model.half(config)
    # Model = save_model(config)
    # Model.half()
    # Model.to(device)

    Model = model(config).to(device)
    loss = nn.L1Loss(reduction="sum").to(device)
    optimizer_model = torch.optim.Adam(Model.parameters(), lr=learning_rate, eps=1e-3)
    scheduler_model = CosineAnnealingWarmRestarts(optimizer_model, T_0=5, T_mult=2)

    writer = SummaryWriter("logs_train")
    total_train_step = 0
    total_test_step = 0
    total_step_all = 0
    for i in range(epochs):
        print("------第{}轮训练开始------".format(i + 1))
        labels_tr_list = []
        y_hat_tr_list = []
        total_tr_loss = 0
        Model.train()
        for a, item in enumerate(dataloader_tr):
            snp_tr, mRNA_tr, mCG_tr, mCHG_tr, mCHH_tr, labels_tr = item
            snp_tr, mRNA_tr, mCG_tr, mCHG_tr, mCHH_tr, labels_tr = \
                snp_tr.to(device), mRNA_tr.to(device), mCG_tr.to(device), mCHG_tr.to(device), mCHH_tr.to(device), labels_tr.to(device)

            # snp_tr, mRNA_tr, mCG_tr, mCHG_tr, mCHH_tr, labels_tr = \
            #     snp_tr.half(), mRNA_tr.half(), mCG_tr.half(), mCHG_tr.half(), mCHH_tr.half(), labels_tr.half()

            # labels_tr = labels_tr.squeeze(-1)  # 计算交叉熵的label处理
            labels_tr = labels_tr.reshape(-1, 1)

            snp_tr = snp_tr.reshape(snp_tr.shape[0], 1, snp_tr.shape[1])
            snp_dim_output_tr, snp_tr = Model.snp_dim_reduction(snp_tr)  #snp_tr: snp_dimention_reduction

            snp_tr = snp_tr.reshape(snp_tr.shape[0], 1, snp_tr.shape[1])
            mRNA_tr = mRNA_tr.reshape(mRNA_tr.shape[0], 1, mRNA_tr.shape[1])
            mCG_tr = mCG_tr.reshape(mCG_tr.shape[0], 1, mCG_tr.shape[1])
            mCHG_tr = mCHG_tr.reshape(mCHG_tr.shape[0], 1, mCHG_tr.shape[1])
            mCHH_tr = mCHH_tr.reshape(mCHH_tr.shape[0], 1, mCHH_tr.shape[1])

            all_omics_tr = torch.cat((snp_tr, mRNA_tr, mCG_tr, mCHG_tr, mCHH_tr), 2)

            all_output_tr, snp_output_tr, mRNA_output_tr, mCG_output_tr, mCHG_output_tr, mCHH_output_tr, hidden_all_tr, \
            hidden_snp_tr, hidden_mRNA_tr, hidden_mCG_tr, hidden_mCHG_tr, hidden_mCHH_tr = Model(all_omics_tr, snp_tr, mRNA_tr, mCG_tr, mCHG_tr, mCHH_tr)

            hidden_integrate_tr = torch.cat((hidden_all_tr, hidden_snp_tr, hidden_mRNA_tr, hidden_mCG_tr, hidden_mCHG_tr, hidden_mCHH_tr), 1)
            hidden_integrate_tr = hidden_integrate_tr.reshape(hidden_integrate_tr.shape[0], 1, hidden_integrate_tr.shape[1])
            output_integrate_tr = Model.integrate(hidden_integrate_tr)

            l_snp_dim_reduction = loss(snp_dim_output_tr, labels_tr)
            l_all = loss(all_output_tr, labels_tr)
            l_snp = loss(snp_output_tr, labels_tr)
            l_mRNA = loss(mRNA_output_tr, labels_tr)
            l_mCG = loss(mCG_output_tr, labels_tr)
            l_mCHG = loss(mCHG_output_tr, labels_tr)
            l_mCHH = loss(mCHH_output_tr, labels_tr)
            l_integrate = loss(output_integrate_tr, labels_tr)

            total_loss_tr = l_snp_dim_reduction + (l_all+l_snp+l_mRNA+l_mCG+l_mCHG+l_mCHH) + 1 * l_integrate
            total_tr_loss += total_loss_tr.item()  # 计算每一轮的总loss

            # output_integrate_pred = output_integrate_tr.item()
            output_integrate_pred = output_integrate_tr.cpu()
            output_integrate_pred = output_integrate_pred.detach().numpy()
            y_hat_tr_list = y_hat_tr_list + list(output_integrate_pred)
            labels_tr_list.extend(labels_tr.cpu().detach().numpy())

            optimizer_model.zero_grad()
            total_loss_tr.backward()
            optimizer_model.step()

            # total_train_step += 1
            # if total_train_step % 100 == 0:
            #     print("训练次数：{}, loss：{}".format(total_train_step, total_loss_tr.item()))
        scheduler_model.step()
        r = computeCorrelation(y_hat_tr_list, labels_tr_list)
        mae = MAE(y_hat_tr_list, labels_tr_list)
        mse = MSE(y_hat_tr_list, labels_tr_list)

        total_step_all += 1
        print("---train---")
        print("loss of train set:{}".format(total_tr_loss))
        print("R_tr:{}".format(r))
        writer.add_scalar("train_loss", total_tr_loss, total_step_all)
        writer.add_scalar("train_R", r, total_step_all)

        labels_val_list = []
        y_hat_val_list = []
        Model.eval()
        with torch.no_grad():
            total_val_loss = 0

            for a_val, item_val in enumerate(dataloader_val):
                snp_val, mRNA_val, mCG_val, mCHG_val, mCHH_val, labels_val = item_val
                snp_val, mRNA_val, mCG_val, mCHG_val, mCHH_val, labels_val = \
                    snp_val.to(device), mRNA_val.to(device), mCG_val.to(device), mCHG_val.to(device), mCHH_val.to(device), labels_val.to(device)

                # snp_val, mRNA_val, mCG_val, mCHG_val, mCHH_val, labels_val = \
                #     snp_val.half(), mRNA_val.half(), mCG_val.half(), mCHG_val.half(), mCHH_val.half(), labels_val.half()

                labels_val = labels_val.reshape(-1, 1)

                snp_val = snp_val.reshape(snp_val.shape[0], 1, snp_val.shape[1])
                snp_dim_output_val, snp_val = Model.snp_dim_reduction(snp_val)  # snp_tr: snp_dimention_reduction

                snp_val = snp_val.reshape(snp_val.shape[0], 1, snp_val.shape[1])
                mRNA_val = mRNA_val.reshape(mRNA_val.shape[0], 1, mRNA_val.shape[1])
                mCG_val = mCG_val.reshape(mCG_val.shape[0], 1, mCG_val.shape[1])
                mCHG_val = mCHG_val.reshape(mCHG_val.shape[0], 1, mCHG_val.shape[1])
                mCHH_val = mCHH_val.reshape(mCHH_val.shape[0], 1, mCHH_val.shape[1])

                all_omics_val = torch.cat((snp_val, mRNA_val, mCG_val, mCHG_val, mCHH_val), 2)

                all_output_val, snp_output_val, mRNA_output_val, mCG_output_val, mCHG_output_val, mCHH_output_val, hidden_all_val, \
                hidden_snp_val, hidden_mRNA_val, hidden_mCG_val, hidden_mCHG_val, hidden_mCHH_val = Model(all_omics_val, snp_val, mRNA_val, mCG_val, mCHG_val, mCHH_val)

                hidden_integrate_val = torch.cat((hidden_all_val, hidden_snp_val, hidden_mRNA_val, hidden_mCG_val, hidden_mCHG_val, hidden_mCHH_val), 1)
                hidden_integrate_val = hidden_integrate_val.reshape(hidden_integrate_val.shape[0], 1, hidden_integrate_val.shape[1])
                output_integrate_val = Model.integrate(hidden_integrate_val)

                output_integrate_pred_val = output_integrate_val.cpu()
                output_pred_val = output_integrate_pred_val.detach().numpy()
                y_hat_val_list = y_hat_val_list + list(output_pred_val)
                labels_val_list.extend(labels_val.cpu().detach().numpy())

            r_val = computeCorrelation(y_hat_val_list, labels_val_list)
            mae_val = MAE(y_hat_val_list, labels_val_list)
            mse_val = MSE(y_hat_val_list, labels_val_list)

            print("---val---")
            print("R_val:{}".format(r_val))
            print("MAE_val:{}".format(mae_val))
            print("MSE_val:{}".format(mse_val))

            writer.add_scalar("R_val", r_val, total_step_all)
            writer.add_scalar("MAE_val", mae_val, total_step_all)
            writer.add_scalar("MSE_val", mse_val, total_step_all)

            # if i >= 90 and r_val > 0.78:
            #     torch.save(Model.state_dict(), "FT16_model/save_model_Integrate_model{}.pth".format(i + 1))
            #     print("模型已保存")

        Model.eval()
        with torch.no_grad():
            total_te_loss = 0
            labels_te_list = []
            y_hat_te_list = []

            for a, item_te in enumerate(dataloader_te):
                snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te, labels_te = item_te

                # snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te, labels_te = \
                #     snp_te.half(), mRNA_te.half(), mCG_te.half(), mCHG_te.half(), mCHH_te.half(), labels_te.half()

                snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te, labels_te = \
                    snp_te.to(device), mRNA_te.to(device), mCG_te.to(device), mCHG_te.to(device), mCHH_te.to(device), labels_te.to(device)
                labels_te = labels_te.reshape(-1, 1)  # 计算交叉熵的label处理

                snp_te = snp_te.reshape(snp_te.shape[0], 1, snp_te.shape[1])
                snp_dim_output_te, snp_te = Model.snp_dim_reduction(snp_te)  # snp_tr: snp_dimention_reduction

                snp_te = snp_te.reshape(snp_te.shape[0], 1, snp_te.shape[1])
                mRNA_te = mRNA_te.reshape(mRNA_te.shape[0], 1, mRNA_te.shape[1])
                mCG_te = mCG_te.reshape(mCG_te.shape[0], 1, mCG_te.shape[1])
                mCHG_te = mCHG_te.reshape(mCHG_te.shape[0], 1, mCHG_te.shape[1])
                mCHH_te = mCHH_te.reshape(mCHH_te.shape[0], 1, mCHH_te.shape[1])

                all_omics_te = torch.cat((snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te), 2)

                all_output_te, snp_output_te, mRNA_output_te, mCG_output_te, mCHG_output_te, mCHH_output_te, hidden_all_te, \
                hidden_snp_te, hidden_mRNA_te, hidden_mCG_te, hidden_mCHG_te, hidden_mCHH_te = Model(all_omics_te, snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te)

                hidden_integrate_te = torch.cat((hidden_all_te, hidden_snp_te, hidden_mRNA_te, hidden_mCG_te, hidden_mCHG_te, hidden_mCHH_te), 1)
                hidden_integrate_te = hidden_integrate_te.reshape(hidden_integrate_te.shape[0], 1,hidden_integrate_te.shape[1])
                output_integrate_te = Model.integrate(hidden_integrate_te)

                output_integrate_pred = output_integrate_te.cpu()
                output_integrate_pred = output_integrate_pred.detach().numpy()
                y_hat_te_list = y_hat_te_list + list(output_integrate_pred)
                labels_te_list.extend(labels_te.cpu().detach().numpy())

            r_te = computeCorrelation(y_hat_te_list, labels_te_list)
            mae_te = MAE(y_hat_te_list, labels_te_list)
            mse_te = MSE(y_hat_te_list, labels_te_list)

            print("---test---")
            print("R_te:{}".format(r_te))
            print("MAE_te:{}".format(mae_te))
            print("MSE_te:{}".format(mse_te))

            writer.add_scalar("R_te", r_te, total_step_all)
            writer.add_scalar("MAE_te", mae_te, total_step_all)
            writer.add_scalar("MSE_te", mse_te, total_step_all)
        #
        #     # torch.save(Model.state_dict(), "E:/pytorch/TMOIA/LGG/save_modelIntegrate_model{}.pth".format(i + 1))
        #     # print("模型已保存")
    writer.close()

def test(dataloader_te, model_te, config):
    labels_te_list = []
    y_hat_te_list = []
    total_te_loss = 0
    Model = model(config).to(device)
    loss = nn.L1Loss(reduction="sum").to(device)
    Model.load_state_dict(model_te)

    Model.eval()

    for a, item_te in enumerate(dataloader_te):
        snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te, labels_te = item_te
        snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te, labels_te = \
            snp_te.to(device), mRNA_te.to(device), mCG_te.to(device), mCHG_te.to(device), mCHH_te.to(device), labels_te.to(device)
        labels_te = labels_te.reshape(-1, 1)  # 计算交叉熵的label处理

        snp_te = snp_te.reshape(snp_te.shape[0], 1, snp_te.shape[1])
        snp_dim_output_te, snp_te = Model.snp_dim_reduction(snp_te)  # snp_tr: snp_dimention_reduction

        snp_te = snp_te.reshape(snp_te.shape[0], 1, snp_te.shape[1])
        mRNA_te = mRNA_te.reshape(mRNA_te.shape[0], 1, mRNA_te.shape[1])
        mCG_te = mCG_te.reshape(mCG_te.shape[0], 1, mCG_te.shape[1])
        mCHG_te = mCHG_te.reshape(mCHG_te.shape[0], 1, mCHG_te.shape[1])
        mCHH_te = mCHH_te.reshape(mCHH_te.shape[0], 1, mCHH_te.shape[1])

        all_omics_te = torch.cat((snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te), 2)

        all_output_te, snp_output_te, mRNA_output_te, mCG_output_te, mCHG_output_te, mCHH_output_te, hidden_all_te, \
        hidden_snp_te, hidden_mRNA_te, hidden_mCG_te, hidden_mCHG_te, hidden_mCHH_te = Model(all_omics_te, snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te)

        hidden_integrate_te = torch.cat((hidden_all_te, hidden_snp_te, hidden_mRNA_te, hidden_mCG_te, hidden_mCHG_te, hidden_mCHH_te), 1)
        hidden_integrate_te = hidden_integrate_te.reshape(hidden_integrate_te.shape[0], 1,hidden_integrate_te.shape[1])
        output_integrate_te = Model.integrate(hidden_integrate_te)

        l_snp_dim_reduction = loss(snp_dim_output_te, labels_te)
        l_all = loss(all_output_te, labels_te)
        l_snp = loss(snp_output_te, labels_te)
        l_mRNA = loss(mRNA_output_te, labels_te)
        l_mCG = loss(mCG_output_te, labels_te)
        l_mCHG = loss(mCHG_output_te, labels_te)
        l_mCHH = loss(mCHH_output_te, labels_te)
        l_integrate = loss(output_integrate_te, labels_te)

        total_loss_te = l_snp_dim_reduction + (l_all + l_snp + l_mRNA + l_mCG + l_mCHG + l_mCHH) + 1 * l_integrate
        total_te_loss += total_loss_te.item()  # 计算每一轮的总loss

        output_integrate_pred = output_integrate_te.cpu()
        output_integrate_pred = output_integrate_pred.detach().numpy()
        y_hat_te_list = y_hat_te_list + list(output_integrate_pred)
        labels_te_list.extend(labels_te.cpu().detach().numpy())

        # print(l_snp_dim_reduction.item(), l_all.item(), l_snp.item(), l_mRNA.item(), l_mCG.item(), l_mCHG.item(), l_mCHH.item(), l_integrate.item())

    # total_te_loss = abs(total_te_loss - 446.6678485274315)
    # r_te = computeCorrelation(y_hat_te_list, labels_te_list)
    # mae_te = MAE(y_hat_te_list, labels_te_list)
    # mse_te = MSE(y_hat_te_list, labels_te_list)

    return total_te_loss

    # print("---test---")
    # print("R_te:{}".format(r_te))
    # print("MAE_te:{}".format(mae_te))
    # print("MSE_te:{}".format(mse_te))
    # print("loss:{}".format(total_te_loss))

