import os
import random
from config import *
from utils1 import *
from model import *
import torch.utils.data.dataloader as Dataloader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(0)

model_te = torch.load("Ath_model/FT16_model/save_modelIntegrate_model151.pth")

i = "01"

labels_dir = "ath_pheno_std.pure.txt"
snp_dir = os.path.join("Ath_2000feature/omics2000/r_"+ str(i)+ "snp2000.csv")
mRNA_dir = "Ath_2000feature/omics2000/mRNA2000.csv"
mCG_dir = "Ath_2000feature/omics2000/mCG2000.csv"
mCHG_dir = "Ath_2000feature/omics2000/mCHG2000.csv"
mCHH_dir = "Ath_2000feature/omics2000/mCHH2000.csv"

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

mRNA_te = mRNA.loc[id_te]
mCG_te = mCG.loc[id_te]
mCHG_te = mCHG.loc[id_te]
mCHH_te = mCHH.loc[id_te]
labels_te = labels.loc[id_te]
snp_te = snp.loc[id_te]

dataset_te = dataset(snp_te, mRNA_te, mCG_te, mCHG_te, mCHH_te, labels_te)
dataloader_te = Dataloader.DataLoader(dataset_te, batch_size=1, shuffle=True)

test(dataloader_te, model_te, config)