import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data.dataloader as Dataloader
import os
from utils1 import *
from config import *

i = "01"

labels_dir = "ath_pheno_std.pure.txt"
snp_dir = os.path.join("Ath_2000feature/omics2000/r_"+ str(i)+ "snp2000.csv")
mRNA_dir = "Ath_2000feature/omics2000/mRNA2000.csv"
mCG_dir = "Ath_2000feature/omics2000/mCG2000.csv"
mCHG_dir = "Ath_2000feature/omics2000/mCHG2000.csv"
mCHH_dir = "Ath_2000feature/omics2000/mCHH2000.csv"

id_tr = pd.read_table(os.path.join("Ath_dataset/ath_THvar0.25_gene_600_FT16_AllOmics/r10/"+str(i)+"_trn_ath_ids.txt"), header=None)
id_tr = id_tr[0].to_list()
id_val = pd.read_table(os.path.join("Ath_dataset/ath_THvar0.25_gene_600_FT16_AllOmics/r10/"+str(i)+"_val_ath_ids.txt"), header=None)
id_val = id_val[0].to_list()
id_te = pd.read_table(os.path.join("Ath_dataset/ath_THvar0.25_gene_600_FT16_AllOmics/r10/"+str(i)+"_tst_ath_ids.txt"), header=None)
id_te = id_te[0].to_list()

mRNA = pd.read_csv(mRNA_dir, index_col=0)
mCG = pd.read_csv(mCG_dir, index_col=0)
mCHG = pd.read_csv(mCHG_dir, index_col=0)
mCHH = pd.read_csv(mCHH_dir, index_col=0)
snp = pd.read_csv(snp_dir, index_col=0)
labels = pd.read_table(labels_dir, header=None, index_col=0)
# labels = pd.read_csv(labels, index_col=0)

# snp, mRNA, meth, mCHG, mCHH, labels = input_data(snp_dir, mRNA_dir, mCG_dir, mCHG_dir, mCHH_dir, labels_dir, 2000)

mRNA_tr, mRNA_val, mRNA_te = mRNA.loc[id_tr], mRNA.loc[id_val], mRNA.loc[id_te]
mCG_tr, mCG_val, mCG_te = mCG.loc[id_tr], mCG.loc[id_val], mCG.loc[id_te]
mCHG_tr, mCHG_val, mCHG_te = mCHG.loc[id_tr], mCHG.loc[id_val], mCHG.loc[id_te]
mCHH_tr, mCHH_val, mCHH_te = mCHH.loc[id_tr], mCHH.loc[id_val], mCHH.loc[id_te]
labels_tr, labels_val, labels_te = labels.loc[id_tr], labels.loc[id_val], labels.loc[id_te]
snp_tr, snp_val, snp_te = snp.loc[id_tr], snp.loc[id_val], snp.loc[id_te]

dataset_tr = dataset(snp_tr, mRNA_tr, mCG_tr, mCHG_tr, mCHH_tr, labels_tr)
dataset_val = dataset(snp_val, mRNA_val, mCG_val, mCHG_val, mCHH_val, labels_val)
dataset_te = dataset(snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te, labels_te)

dataloader_tr = Dataloader.DataLoader(dataset_tr, batch_size=128, shuffle=True)
dataloader_val = Dataloader.DataLoader(dataset_val, batch_size=1, shuffle=True)
dataloader_te = Dataloader.DataLoader(dataset_te, batch_size=1, shuffle=True)

train_val(config, config["epochs"], config["learning_rate"], dataloader_tr, dataloader_val, dataloader_te)

