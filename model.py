import torch
import torch.nn as nn
class CRNN (nn.Module):
    def __init__(self, input_height, num_classes):
        super(CRNN, self).__init__()
        self.filter_size = 64
        #self.image_size = input_height // 4
        self.image_size = input_height // input_height
        self.hidden_size = 256
        self.lstm_hidden = 256
        self.lstm_layer = 2
        self.attention_head = 8

        self.conv1 = nn.Conv2d(1, self.filter_size, 3, 1, 1)
        self.conv12 = nn.Conv2d(self.filter_size, self.filter_size, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.filter_size)
        self.bn12 = nn.BatchNorm2d(self.filter_size)
        self.conv2 = nn.Conv2d(self.filter_size, self.filter_size * 2, 3, 1, 1)
        self.conv22 = nn.Conv2d(self.filter_size *2, self.filter_size*2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.filter_size * 2)
        self.bn22 = nn.BatchNorm2d(self.filter_size*2)
        self.conv23 = nn.Conv2d(self.filter_size * 2, self.filter_size * 4, 3, 1, 1)
        self.bn23 = nn.BatchNorm2d(self.filter_size * 4)
        self.conv24 = nn.Conv2d(self.filter_size * 4, self.filter_size * 4, 3, 1, 1)
        self.bn24 = nn.BatchNorm2d(self.filter_size * 4)
        self.conv25 = nn.Conv2d(self.filter_size * 4, self.filter_size * 4, 4, 1, 0)
        self.bn25 = nn.BatchNorm2d(self.filter_size * 4)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        #self.fc1 = nn.Linear(self.filter_size * 2 * self.image_size, self.hidden_size)
        self.conv3 = nn.Conv1d(self.filter_size * 4 * self.image_size, self.hidden_size, kernel_size=1)

        self.lstm1 = nn.LSTM(self.hidden_size, self.lstm_hidden, num_layers=self.lstm_layer, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(self.lstm_hidden*2, self.lstm_hidden, num_layers=self.lstm_layer, bidirectional= True, batch_first=True)

        self.attention = nn.MultiheadAttention(self.lstm_hidden*2, self.attention_head, batch_first=True)

        self.layer_norm1 = nn.LayerNorm(self.lstm_hidden*2)
        self.layer_norm2 = nn.LayerNorm(self.lstm_hidden*2)
        self.fc2 = nn.Linear(self.lstm_hidden*2, num_classes+1)

    def forward(self, x):

        output = self.conv1(x)        #(B C H W)  #(B 1 H W) --> #(B 128 H W)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv12(output)   #(B 128 H W) --> #(B 128 H W)
        output = self.bn12(output)
        output = self.relu(output)

        output = self.pool(output)    #(B 128 H/2 W/2) --> #(B 128 H/2 W/2)

        output = self.conv2(output)   #(B 128 H/2 W/2) --> #(B 256 H/2 W/2)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv22(output)  #(B 256 H/2 W/2) --> #(B 256 H/2 W/2)
        output = self.bn22(output)
        output = self.relu(output)

        output = self.pool(output) #(B 256 H/4 W/4) --> #(B 256 H/4 W/4)

        output = self.conv23(output)
        output = self.bn23(output)
        output = self.relu(output)

        output = self.conv24(output)
        output = self.bn24(output)
        output = self.relu(output)

        output = self.pool(output)
        output = self.conv25(output)
        output = self.bn25(output)
        output = self.relu(output)
        bs, c, h, w = output.size()
        output = output.permute(0, 1, 3, 2) #(B 256 W/4 H/4)
        output = output.reshape(bs, c*h, w) #(B (256 * H/4) w/4)


        output = self.conv3(output) #(B 256 w/4)
        output = output.permute(0, 2, 1) #(B W/4 256)
        output1, _ = self.lstm1(output)
        output1 = self.layer_norm1(output1)

        attention_out, _ = self.attention(output1, output1, output1)

        output2, _ = self.lstm2(attention_out)
        output2 = self.layer_norm2(output2)

        output = output1 + attention_out + output2
        output = self.fc2(output)
        return output