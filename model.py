import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, image_size = 32, num_classes = 10, in_channel = 3, out_channels = (16,32,64),
                 fc_out = (4096, 1024), kernel = 3, pad = 1, stride = 1, pool_kernel = 2, pool_stride = 2,
                 batch_norm=False, drop_out=0, dense=False):
        super().__init__()
        self.image_size = image_size
        self.batch_norm = batch_norm
        self.drop_out = drop_out
        self.kernel = kernel
        self.pad = pad
        self.stride = stride
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.dense = dense

        c1_out, c2_out, c3_out = out_channels

        self.conv_layer1 = nn.Conv2d(in_channel,c1_out,kernel, stride, pad)
        c2_in = c1_out + in_channel if dense else c1_out

        self.conv_layer2 = nn.Conv2d(c2_in, c2_out,kernel, stride, pad)
        c3_in = c2_out + c2_in if dense else c2_out

        self.conv_layer3 = nn.Conv2d(c3_in, c3_out,kernel, stride, pad)
        fc_in_ch = c3_out

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(c1_out)
            self.batch_norm2 = nn.BatchNorm2d(c2_out)
            self.batch_norm3 = nn.BatchNorm2d(c3_out)
            self.batch_norm4 = nn.BatchNorm1d(fc_out[0])
            self.batch_norm5 = nn.BatchNorm1d(fc_out[1])

        output_image_size = self._calculate_image_size()
        self.fc1_input_size = fc_in_ch * output_image_size**2
        self.fc1 = nn.Linear(self.fc1_input_size, out_features=fc_out[0])
        self.drop_out1 = nn.Dropout(self.drop_out)
        self.fc2 = nn.Linear(fc_out[0], out_features=fc_out[1])
        self.drop_out2 = nn.Dropout(self.drop_out)
        self.fc3 = nn.Linear(fc_out[1], out_features=num_classes)


    def forward(self, x):

        c1 = self.conv_layer1(x)
        if self.batch_norm:
            c1 = self.batch_norm1(c1)
        a1 = F.relu(c1)
        if not self.dense:
            mp1 = F.max_pool2d(a1, kernel_size = self.pool_kernel, stride=self.pool_stride)
        else:
            mp1 = a1

        in2 = torch.cat([x, mp1],1) if self.dense else mp1
        c2 = self.conv_layer2(in2)
        if self.batch_norm:
            c2 = self.batch_norm2(c2)
        a2 = F.relu(c2)
        if not self.dense:
            mp2 = F.max_pool2d(a2, kernel_size = self.pool_kernel, stride=self.pool_stride)
        else:
            mp2 = a2

        in3 = torch.cat([x, mp1, mp2],1) if self.dense else mp2
        c3 = self.conv_layer3(in3)
        if self.batch_norm:
            c3 = self.batch_norm3(c3)
        a3 = F.relu(c3)
        mp3 = F.max_pool2d(a3, kernel_size = self.pool_kernel, stride=self.pool_stride)

        in_fc = mp3

        in_fc = in_fc.view(-1, self.fc1_input_size)
        fc1 = self.fc1(in_fc)
        if self.batch_norm:
            fc1 = self.batch_norm4(fc1)
        fca1 = F.relu(fc1)
        fca1 = self.drop_out1(fca1)
        fc2 = self.fc2(fca1)
        if self.batch_norm:
            fc = self.batch_norm5(fc2)
        fca2 = F.relu(fc2)
        fca2 = self.drop_out2(fca2)
        fc3 = self.fc3(fca2)

        return fc3

    def _calculate_image_size(self):
         #For convolution 1
        output_image_size = (self.image_size + 2*self.pad - self.kernel)//self.stride + 1
        # For pooling 1
        if not self.dense:
            output_image_size = (output_image_size - self.pool_kernel)//self.pool_stride + 1

        #For convolution 2
        output_image_size = (output_image_size + 2*self.pad - self.kernel)//self.stride + 1
        # For pooling 2
        if not self.dense:
            output_image_size = (output_image_size - self.pool_kernel)//self.pool_stride + 1

        #For convolution 3
        output_image_size = (output_image_size + 2*self.pad - self.kernel)//self.stride + 1
        # For pooling 3
        output_image_size = (output_image_size - self.pool_kernel)//self.pool_stride + 1

        return output_image_size
