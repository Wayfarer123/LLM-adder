from IPython.display import clear_output
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
device = torch.device('cuda')
sns.set_style('darkgrid')


class Model(nn.Module):
    def __init__(self, network, tokenizer, max_len=512, base_model=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.base_model = base_model

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def generate(self, a, b):
        input_str = self.tokenizer([' '.join(list(str(a))) + ' + ' + ' '.join(list(str(b))) + ' = '],
                                   return_tensors='pt')
        generator = self.network if self.base_model else self.network.base_model

        if self.max_len is not None:
            out = generator.generate(input_str['input_ids'].to(device), max_length=self.max_len)
        else:
            out = generator.generate(input_str['input_ids'].to(device))
        return ''.join(self.tokenizer.batch_decode(out,
                                skip_special_tokens=True)[0].split(' '))

    def train(self, *args, **kwargs):
        train_model(self.network, *args, **kwargs)


def train_model(model, train_loader,
                optimizer, num_epochs,
                scheduler=None,
                store_path=None,
                init_weights_path=None,
                device='cuda'):
    loss_by_iter = []

    if init_weights_path is not None:
        model.load_state_dict(torch.load(init_weights_path))

    for epoch in tqdm(range(num_epochs), desc=f"Training progress", colour="#00ff00"):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        model.train()

        for i_step, (inp, attention_mask, target) in enumerate(tqdm(train_loader, leave=False,
                                                                    desc=f"Epoch {epoch + 1}/{num_epochs}",
                                                                    colour="#005500")):

            model.zero_grad()
            inp, attention_mask, target = inp.to(device), attention_mask.to(device), target.to(device)

            out = model(input_ids=inp, attention_mask=attention_mask,
                        labels=target)

            loss_value = out[0]
            loss_value.backward()

            optimizer.step()

            loss_by_iter.append(loss_value.detach().cpu())

            if (i_step + 1) % 100 == 0:
                if scheduler is not None:
                    scheduler.step()

                clear_output(True)
                plot_progress(loss_by_iter)

                plt.show()

        gc.collect()
        torch.cuda.empty_cache()

        clear_output(True)

        plot_progress(loss_by_iter)
        plt.show()

        if store_path is not None:
            torch.save(model.state_dict(), store_path)


def plot_progress(loss, ax=None, title="Loss on train"):
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

    ax.plot(np.arange(len(loss)), loss, lw=3,
            color=sns.color_palette()[0], )

    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel('Loss', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)

    ax.set_title(title, fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
