import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as Dateset
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Multi_Head_Attention(nn.Module):
	# '''
	# params: dim_model-->hidden dim      num_head
	# '''
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0    # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head   # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:
        #     mask = mask.repeat(self.num_head, 1, 1)
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale) # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head) # reshape 回原来的形状
        out = self.fc(context)   # 全连接
        out = self.dropout(out)
        out = out + x      # 残差连接,ADD
        out = self.layer_norm(out)
        return out

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class Transformer_mRNA(nn.Module):
    def __init__(self, config):
        super(Transformer_mRNA, self).__init__()
        # self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config["Transformer_mRNA_input_dim"], config["Transformer_mRNA_head"], config["Transformer_mRNA_hidden"], config["Transformer_drop"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # for _ in range(config["Transformer_num_encoder"])])# 多次Encoder
            for _ in range(config["Transformer_num_encoder"])])

        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        self.fc1 = nn.Linear(config["Transformer_mRNA_input_dim"], 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, config["Transformer_output"])

    def forward(self, x):
        # out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(x)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out, h_out

class Transformer_mCG(nn.Module):
    def __init__(self, config):
        super(Transformer_mCG, self).__init__()
        # self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config["Transformer_mCG_input_dim"], config["Transformer_mCG_head"], config["Transformer_mCG_hidden"], config["Transformer_drop"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # for _ in range(config["Transformer_num_encoder"])])# 多次Encoder
            for _ in range(config["Transformer_num_encoder"])])

        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        self.fc1 = nn.Linear(config["Transformer_mCG_input_dim"], 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, config["Transformer_output"])

    def forward(self, x):
        # out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(x)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out, h_out

class Transformer_mCHG(nn.Module):
    def __init__(self, config):
        super(Transformer_mCHG, self).__init__()
        # self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config["Transformer_mCHG_input_dim"], config["Transformer_mCHG_head"], config["Transformer_mCHG_hidden"], config["Transformer_drop"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # for _ in range(config["Transformer_num_encoder"])])# 多次Encoder
            for _ in range(config["Transformer_num_encoder"])])

        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        self.fc1 = nn.Linear(config["Transformer_mCHG_input_dim"], 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, config["Transformer_output"])

    def forward(self, x):
        # out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(x)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out, h_out

class Transformer_mCHH(nn.Module):
    def __init__(self, config):
        super(Transformer_mCHH, self).__init__()
        # self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config["Transformer_mCHH_input_dim"], config["Transformer_mCHH_head"], config["Transformer_mCHH_hidden"], config["Transformer_drop"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # for _ in range(config["Transformer_num_encoder"])])# 多次Encoder
            for _ in range(config["Transformer_num_encoder"])])

        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        self.fc1 = nn.Linear(config["Transformer_mCHH_input_dim"], 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, config["Transformer_output"])

    def forward(self, x):
        # out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(x)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out, h_out

class Transformer_snp(nn.Module):
    def __init__(self, config):
        super(Transformer_snp, self).__init__()
        # self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config["Transformer_snp_input_dim"], config["Transformer_snp_head"], config["Transformer_snp_hidden"], config["Transformer_drop"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # for _ in range(config["Transformer_num_encoder"])])# 多次Encoder
            for _ in range(config["Transformer_num_encoder"])])

        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        self.fc1 = nn.Linear(config["Transformer_snp_input_dim"], 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, config["Transformer_output"])

    def forward(self, x):
        # out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(x)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out, h_out

class snp_dimention_reduction(nn.Module):
    def __init__(self, config):
        super(snp_dimention_reduction, self).__init__()
        self.fc1 = nn.Linear(config["snp_input_dim"], 2000)
        self.fc2 = nn.Linear(2000, 200)
        self.fc3 = nn.Linear(200, config["Transformer_output"])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        snp_out = self.fc1(x)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out, snp_out

class Transformer_all(nn.Module):
    def __init__(self, config):
        super(Transformer_all, self).__init__()
        # self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config["Transformer_all_input_dim"], config["Transformer_all_head"], config["Transformer_all_hidden"], config["Transformer_drop"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # for _ in range(config["Transformer_num_encoder"])])   # 多次Encoder
            for _ in range(config["Transformer_num_encoder"])])

        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        self.fc1 = nn.Linear(config["Transformer_all_input_dim"], 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, config["Transformer_output"])

    def forward(self, x):
        # out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(x)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out, h_out

class Transformer_Integrate(nn.Module):
    def __init__(self, config):
        super(Transformer_Integrate, self).__init__()
        # self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config["Transformer_integrate_input_dim"], config["Transformer_integrate_head"], config["Transformer_integrate_hidden"], config["Transformer_drop"])
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # for _ in range(config["Transformer_num_encoder"])])   # 多次Encoder
            for _ in range(config["Transformer_num_encoder"])])
        # self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        self.fc1 = nn.Linear(config["Transformer_integrate_input_dim"], 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, config["Transformer_output"])

    def forward(self, x):
        # out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(x)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        h_out = F.relu(self.fc1(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class model(nn.Module):
    def __init__(self, config):
        super(model, self).__init__()

        self.net_mRNA = Transformer_mRNA(config).to(device)
        self.net_mCG = Transformer_mCG(config).to(device)
        self.net_mCHG = Transformer_mCHG(config).to(device)
        self.net_mCHH = Transformer_mCHH(config).to(device)
        self.net_snp = Transformer_snp(config).to(device)
        self.net_all = Transformer_all(config).to(device)
        self.net_integrate = Transformer_Integrate(config).to(device)
        self.net_snp_dim_reduction = snp_dimention_reduction(config).to(device)

    def forward(self, all, snp, mRNA, mCG, mCHG, mCHH):

        all_output, hidden_all = self.net_all(all)
        snp_output, hidden_snp = self.net_snp(snp)
        mRNA_output, hidden_mRNA = self.net_mRNA(mRNA)
        mCG_output, hidden_mCG = self.net_mCG(mCG)
        mCHG_output, hidden_mCHG = self.net_mCHG(mCHG)
        mCHH_output, hidden_mCHH = self.net_mCHH(mCHH)

        return all_output, snp_output, mRNA_output, mCG_output, mCHG_output, mCHH_output,\
                hidden_all, hidden_snp, hidden_mRNA, hidden_mCG, hidden_mCHG, hidden_mCHH

    def integrate(self, hidden_integrate):
        # hidden_integrate = torch.cat((hidden_all, hidden_mRNA, hidden_meth, hidden_miRNA), 1)
        output_integrate = self.net_integrate(hidden_integrate)

        return output_integrate

    def snp_dim_reduction(self, snp_dim):
        output, hidden_out = self.net_snp_dim_reduction(snp_dim)

        return output, hidden_out