import matplotlib.pyplot as plt
from neuralop.datasets import load_darcy_flow_small   

train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=100, batch_size=4,
    test_resolutions=[16,32], n_tests=[50,50], test_batch_sizes=[4,2],
    )

# train_loader is just wraped torch.utils.data.DataLoader
train_dataset = train_loader.dataset

for res, test_loader in test_loaders.items():
    print(res)
    batch = next(iter(test_loader))
    x = batch['x']
    y = batch['y']

    print(f'Testing samples for res {res} have shape {x.shape[1:]}')

data = train_dataset[0]
x = data['x']
y = data['y']

print(f'Training samples have shape {x.shape[1:]}')

index = 0

data = train_dataset[index]
data = data_processor.preprocess(data, batched=False)
x = data ['x']
y = data['y']
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(2, 2, 1)
ax.imshow(x[0], cmap='gray')
ax.set_title('input x')
ax = fig.add_subplot(2, 2, 2)
ax.imshow(y.squeeze())
ax.set_title('input y')
ax = fig.add_subplot(2, 2, 3)
ax.imshow(x[1])
ax.set_title('x: 1st pos embedding')
ax = fig.add_subplot(2, 2, 4)
ax.imshow(x[2])
ax.set_title('x: 2nd pos embedding')
fig.suptitle('Visualizing one input sample', y=0.98)
plt.tight_layout()
fig.savefig('darcy_flow_sample.png')

