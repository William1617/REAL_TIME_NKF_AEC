import torch
import torch.nn as nn



class ComplexGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bias=True, dropout=0):
        super().__init__()
        self.gru_r = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=False)
        self.gru_i = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=False)

    def forward(self, x_feature, h_rr,h_ir, h_ri, h_ii):
        x_real=x_feature[:,:1,:]
        x_imag=x_feature[:,1:,:]
        Frr, h_rr = self.gru_r(x_real, h_rr)
        Fir, h_ir = self.gru_r(x_imag, h_ir)
        Fri, h_ri = self.gru_i(x_real, h_ri)
        Fii, h_ii = self.gru_i(x_imag, h_ii)
        y_real = Frr - Fii
        y_imag= Fri + Fir
        return torch.cat([y_real,y_imag],dim=1), h_rr, h_ir, h_ri, h_ii
    


class ComplexDense(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.linear_real = nn.Linear(in_channel, out_channel, bias=bias)
        self.linear_imag = nn.Linear(in_channel, out_channel, bias=bias)

    def forward(self, x_feature):
        
        x_real=x_feature[:,:1,:]
        x_imag=x_feature[:,1:,:]
        
        y_real = self.linear_real(x_real)
        y_imag = self.linear_imag(x_imag)
        
        return torch.cat([y_real,y_imag],dim=1)


class ComplexPReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()

    def forward(self, x_feature):

        x_real=x_feature[:,:1,:]
        x_imag=x_feature[:,1:,:]
        y_real=self.prelu(x_real)
        y_imag=self.prelu(x_imag)
        return torch.cat([y_real,y_imag],dim=1)
    

class KGNet(nn.Module):
    def __init__(self, L, fc_dim, rnn_layers, rnn_dim):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.rnn_dim = rnn_dim

        self.fc_in = nn.Sequential(
            ComplexDense(2 * self.L + 1, fc_dim, bias=True),
            ComplexPReLU()
        )

        self.complex_gru = ComplexGRU(fc_dim, rnn_dim, rnn_layers)

        self.fc_out = nn.Sequential(
            ComplexDense(rnn_dim, fc_dim, bias=True),
            ComplexPReLU(),
            ComplexDense(fc_dim, self.L, bias=True)
        )



    def forward(self, input_real,in_imag,h_rr,h_ir,h_ri,h_ii):
        input_featre=torch.cat([input_real,in_imag],dim=1)
        out_feat = self.fc_in(input_featre)
        rnn_outfeat, out_hrr, outh_ir, outh_ri, outh_ii = self.complex_gru(out_feat, h_rr, h_ir, h_ri, h_ii)
        kg_outfeat = self.fc_out(rnn_outfeat)
        return kg_outfeat[:,:1,:].permute(0, 2, 1),kg_outfeat[:,1:,:].permute(0, 2, 1),out_hrr,outh_ir,outh_ri,outh_ii


class NKF(nn.Module):
    def __init__(self, L=4):
        super().__init__()
        self.L = L
        self.kg_net = KGNet(L=self.L, fc_dim=18, rnn_layers=1, rnn_dim=18)

    def forward(self, in_real, in_imag,h_rr,h_ir,h_ri,h_ii):
  
        kg_real,kg_imag,out_hrr,outh_ir,outh_ri,outh_ii = self.kg_net(in_real, in_imag,h_rr,h_ir,h_ri,h_ii)

        return  kg_real,kg_imag,out_hrr,outh_ir,outh_ri,outh_ii


model = NKF(L=4)
model.load_state_dict(torch.load('./nkf_epoch70.pt'), strict=True)

in_real=torch.randn(513,1,9)

in_imag=torch.randn(513,1,9)

h_rr=torch.randn(1,513,18)
h_ir=torch.randn(1,513,18)
h_ri=torch.randn(1,513,18)
h_ii=torch.randn(1,513,18)

torch.onnx.export(model,(in_real,in_imag,h_rr,h_ir,h_ri,h_ii),'./nkf.onnx',
                  input_names = ['in_real', 'in_imag', 'in_hrr', 'in_hir', 'in_hri', 'in_hii'],
                  output_names = ['enh_real', 'enh_imag', 'out_hrr', 'out_hir', 'out_hri', 'out_hii'],
                  opset_version=12)