
import torch
from torchsummary import summary

from nets.gatector import GaTectorBody

if __name__ == "__main__":
    train_mode=0
    if train_mode==0:
        anchor_mask=[[0,1,2]]
    if train_mode == 1:
        anchor_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = GaTectorBody(anchor_mask, 80,train_mode).to(device)
    summary(m, input_size=(3, 416, 416))
