import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class RNN_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out


class RNN_LSTM_EMBED(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM_EMBED, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.lstm1 = nn.LSTMCell(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.embed = nn.Embedding(num_classes, embed_size)
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        x_embed = self.embed(x.view(x.size()[0] * x.size()[1], 1).long())
        x_embed = x_embed.view(x.size()[0], x.size()[1], self.embed_size)

        for i, input_t in enumerate(x_embed.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)

        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out



class cond_RNN_LSTM_embed(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, num_layers, num_labels, num_classes):
        super(cond_RNN_LSTM_embed, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.num_labels = num_labels
        self.lstm1 = nn.LSTMCell(embed_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.embed_x = nn.Embedding(num_classes, embed_size)
        self.embed_x.weight.data.uniform_(-0.1, 0.1)    
        self.embed_label = nn.Embedding(num_labels, embed_size)
        self.embed_label.weight.data.uniform_(-0.1, 0.1)


    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        x_x = x[:,:,0].contiguous()
        x_label = x[:,:,1].contiguous()

        
        x_embed = self.embed_x(x_x.view(x_x.size()[0] * x_x.size()[1], 1).long())
        x_embed = x_embed.view(x_x.size()[0], x_x.size()[1], self.embed_size)
        x_label_embed = self.embed_label(x_label.view(x_label.size()[0] * x_label.size()[1], 1).long())
        x_label_embed = x_label_embed.view(x_label.size()[0], x_label.size()[1], self.embed_size)
        x_all_ = torch.cat((x_embed, x_label_embed), dim = -1) 

        for i, input_t in enumerate(x_all_.chunk(x_all_.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)

        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out



class RNN_LSTM_embed_twin(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, num_layers, num_classes, reverse=False):
        super(RNN_LSTM_embed_twin, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(embed_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.reverse = reverse
        self.embed_size = embed_size
        self.embed = nn.Embedding(num_classes, embed_size)
        self.embed.weight.data.uniform_(-0.1, 0.1)

        if not self.reverse:
            self.ln_hidden = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        outputs = []
        states = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        x_embed = self.embed(x.view(x.size()[0] * x.size()[1], 1).long())
        x_embed = x_embed.view(x.size()[0], x.size()[1], self.embed_size)
        
        for i, input_t in enumerate(x_embed.chunk(x_embed.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
            states  += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        states = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)
        if not self.reverse:
            states_shp = states.size()
            states_reshp = states.view(states_shp[0] * states_shp[1], states_shp[2])
            affine_states = self.ln_hidden(states_reshp)
            states = affine_states.view(states_shp[0], states_shp[1], states_shp[2])

        return (out, states)




class RNN_LSTM_twin(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes, reverse=False):
        super(RNN_LSTM_twin, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.reverse = reverse
        if not self.reverse:
            self.ln_hidden = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x):
        outputs = []
        states = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [h_t]
            states  += [h_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        states = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)
        if not self.reverse:
            states_shp = states.size()
            states_reshp = states.view(states_shp[0] * states_shp[1], states_shp[2])
            affine_states = self.ln_hidden(states_reshp)
            states = affine_states.view(states_shp[0], states_shp[1], states_shp[2])

        return (out, states)




class TWINNET_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TWINNET_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(input_size, hidden_size)
        self.lstm4 = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.back_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        forward_outputs = []
        back_outputs = []

        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        h_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())

        back_h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        back_c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        back_h_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        back_c_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())


        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            forward_outputs += [c_t2]
       
        for back_i, back_input_t in reversed(list(enumerate(x.chunk(x.size(1), dim=1)))):
            back_input_t = back_input_t.contiguous().view(back_input_t.size()[0], 1)
            back_h_t, back_c_t = self.lstm3(back_input_t, (back_h_t, back_c_t))
            back_h_t2, back_c_t2 = self.lstm4(back_c_t, (back_h_t2, back_c_t2))
            back_outputs += [back_c_t2]

        forward_outputs = torch.stack(forward_outputs, 1).squeeze(2)
        back_outputs = torch.stack(back_outputs, 1).squeeze(2)
        shp= (forward_outputs.size()[0], forward_outputs.size()[1])
        forward_out = forward_outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        forward_out = self.fc(forward_out)
        forward_out = forward_out.view(shp[0], shp[1], self.num_classes)
        
        back_out = back_outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        back_out = self.back_fc(back_out)
        back_out = back_out.view(shp[0], shp[1], self.num_classes)


        return [forward_out, back_out]


