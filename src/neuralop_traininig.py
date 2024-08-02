import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop.training import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = 'cuda'

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000, batch_size=32,
    test_resolutions=[16,32], n_tests=[100,50], test_batch_sizes=[32,32],
    positional_encoding=True
    )
data_processor = data_processor.to(device)

model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='Tucker', rank=0.42)
model = model.to(device)

n_params = count_model_params(model)
print(f'TFNO model has {n_params} parameters')
sys.stdout.flush()

optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {'h1' : h1loss, 'l2' : l2loss}

print('\n### Model ###\n', model)
print('\n### Optimizer ###\n', optimizer)
print('\n### Scheduler ###\n', scheduler)
print('\n### Train Loss ###\n', train_loss)
print('\n### Eval Losses ###\n', eval_losses)
sys.stdout.flush()

trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)

trainer.train(train_loader=train_loader,
              test_loaders={},
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)


test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index+5]
    data = data_processor.preprocess(data,batched=False)
    x = data['x']
    y = data['y']
    out = model(x.unsqueeze(0).to(device)).squeeze(0)  

    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x[0].cpu().numpy(), cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], []) 
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y.squeeze().cpu().numpy())
    if index == 0:
        ax.set_title('Ground Truth y')  
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.squeeze().detach().cpu().numpy())
    if index == 0:
        ax.set_title('Predicted y')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truths, and predictions', y=0.98)
plt.tight_layout()
fig.savefig('darcy_flow_small.png') 
