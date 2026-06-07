from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,activation=F.silu,bias=False,dropout=0.1,device=None,dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        out_features = out_features if out_features is not None else in_features
        hidden_features = (hidden_features if hidden_features is not None else int(8 * in_features / 3))

        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)
        self.drop = dropout
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.dropout(y)
        y = self.fc2(y)
        return y