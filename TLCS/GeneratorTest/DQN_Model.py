import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
ACTIONS_PER_AGENT = 8

class LossFuc(nn.Module):
    def __init__(self):
        super(LossFuc, self).__init__()

    def forward(self, x1, y1, x2, y2):  # [bs,num_class]  CE=q*-log(p), q*log(1-p),p=softmax(logits)
        loss1 = torch.nn.functional.mse_loss(x1, y1)
        loss2 = torch.nn.functional.mse_loss(x2, y2)
        return loss1 + loss2

class DQNModel(nn.Module):
    def __init__(self):
        super(DQNModel, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=4, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
            nn.Flatten()
        )
        self.net2 = nn.Sequential(
            nn.Linear(in_features=704, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=ACTIONS_PER_AGENT)
        )
        self.net3 = nn.Sequential(
            nn.Linear(in_features=704, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=4),
        )
        self.loss_fn = LossFuc()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = 'cuda'
        self.to(self.device)

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True).to(self.device)
        # print(f"input:{x.shape}")
        x = x.permute(0, 3, 1, 2)  # Reorder axes to [batch_size, channel, height, width]
        state_conv = self.net1(x)
        # print(f"state_conv:{state_conv.shape}")
        phase_action = self.net2(state_conv)
        # print(f"phase_state:{phase_action.shape}")
        yellow_action = self.net3(state_conv)
        # print(f"yellow_state:{yellow_action.shape}")
        return phase_action, yellow_action

    def get_action(self, x):
        '''
        获取当前动作,返回动作值 phase_action 0-7, yellow_action 0-3
        '''
        phase_action, yellow_action = self.forward(x)
        phase_action = torch.argmax(phase_action, dim=1)
        yellow_action = torch.argmax(yellow_action, dim=1)
        return phase_action, yellow_action

    def get_all_action(self, x):
        '''
        获取当前动作,返回所有动作的得分
        '''
        phase_action, yellow_action = self.forward(x)
        return phase_action, yellow_action

    def train(self, x, y1, y2):
        '''
        训练模型
        param x: 输入状态
        param y1: 目标动作
        param y2: 目标动作
        '''
        phase_action, yellow_action = self.forward(x)
        loss = self.loss_fn(phase_action, y1, yellow_action, y2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

# import timeit
# start_time = timeit.default_timer()
# model = DQNModel()
#
# model.to(device)
# batch_size = 500
# target1 = torch.randint(0, 8, (batch_size,)).to(device)
# target2 = torch.randint(0, 4, (batch_size,)).to(device)
# state_input = torch.randn(batch_size, 12, 50, 2)  # Sample input data with batch size 32
# for i in range(1000):
#     loss = model.train(state_input, target1, target2)
#     # print(loss)
#
# print(f"Time taken: {timeit.default_timer() - start_time}")



# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cuda'
# print(device)
# model = DQNModel()
# model.to(device)
# criterion = LossFuc()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# for i in range(1):
#     batch_size = 2
#     target1 = torch.randint(0, 8, (batch_size,)).to(device)
#     target2 = torch.randint(0, 4, (batch_size,)).to(device)
#     state_input = torch.randn(batch_size, 12, 50, 2).to(device) # Sample input data with batch size 32
#
#     phase_actions, yellow_actions = model(state_input) # Get predicted actions
#     loss = criterion(phase_actions, target1, yellow_actions, target2) # Calculate loss
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     end_time = timeit.default_timer()
# print(f"Time taken: {end_time - start_time}")

# state_input = torch.randn(1, 12, 50, 2).to(device) # Sample input data with batch size 32
# phase_actions, yellow_actions = model.get_action(state_input) # Get predicted actions
# print(phase_actions, yellow_actions)