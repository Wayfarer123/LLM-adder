from random import randint as r
from random import choice
from random import shuffle
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, tokenizer, num_obj=100000, min_len=1,
                 max_len=20, eos='</s>', prompts=True):
        self.num_obj = num_obj
        self.min_len = min_len
        self.max_len = max_len
        self.eos = eos
        self.prompts = prompts
        self.numbers, self.answers = self.generator()
        self.input_lines, self.labels = self.make_input_str()
        self.tokenizer = tokenizer
        self.encode()

    def generator(self):
        # generating random number pairs of random lengths and add them together
        # not the most effective generator, but fast enough to not be a bottleneck

        # choose random lengths for the numbers
        lengths = [r(self.min_len - 1, self.max_len - 1) for i in range(self.num_obj * 2)]
        shuffle(lengths)
        # generate pairs of numbers of chosen lengths
        numbers = [(r(10 ** lengths[2 * i] - 1, 10 ** (lengths[2 * i] + 1)),
                    r(10 ** lengths[2 * i + 1] - 1, 10 ** (lengths[2 * i + 1] + 1)))
                   for i in range(self.num_obj)]
        # calculate correct answers
        answers = [(a + b) for a, b in numbers]

        return numbers, answers

    def make_input_str(self):
        # turn number pairs into an input strings
        input_lines = []
        labels = []

        for (a, b), ans in zip(self.numbers, self.answers):
            # split a and b to digits to be able to tokenize each separately
            label = f'{" ".join(list(str(ans)))} {self.eos}'
            a, b = ' '.join(list(str(a))), ' '.join(list(str(b)))

            if self.prompts:
                # add a random text prompt if necessary
                input_lines.append(choice([
                    f'Calculate the {a} + {b}',
                    f'Find the answer of {a} + {b}',
                    f'{a} + {b} = ',
                    f'What is {a} + {b}',
                    f'What is the sum of {a} and {b}?',
                    f'Add {a} and {b}',
                    f'What is the result of adding {a} and {b}',
                    f'Find the total of {a} and {b}',
                    f'Add {a} and {b} and give the answer',
                    f'Calculate the result of adding {a} and {b}',
                    f'What is the total of {a} and {b}?',
                    f'Find the sum of {a} and {b}',
                    f'Add {a} and {b}',
                    f'Calculate the sum of {a} and {b}',
                    f'What is the result when {a} is added to {b}?',
                    f'Add {a} and {b} and print the answer',
                    f'Calculate the sum of {a} and {b}',
                    f'Find the result of adding {a} and {b}',
                    f'the total of {a} and {b}'
                ]))
            else:
                input_lines.append(f'{a} + {b} = ')

            labels.append(label)

        return input_lines, labels

    def encode(self):
        # apply model pretrained tokenizer on input and target strings
        self.input_lines_encoded = self.tokenizer(self.input_lines,
                                        padding=True,
                                        return_tensors="pt")
        self.labels_encoded = self.tokenizer(self.labels,
                                        padding=True,
                                        return_tensors="pt")

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return self.input_lines_encoded['input_ids'][idx], \
               self.input_lines_encoded['attention_mask'][idx], \
               self.labels_encoded['input_ids'][idx]


class TestDataset(Dataset):
    def __init__(self, num_obj=100000, min_len=1,
                 max_len=20):
        self.num_obj = num_obj
        self.min_len = min_len
        self.max_len = max_len
        self.numbers, self.answers = self.generator()

    def generator(self):
        lengths = [r(self.min_len - 1, self.max_len - 1) for i in range(self.num_obj * 2)]
        shuffle(lengths)
        numbers = [(r(10 ** lengths[2 * i] - 1, 10 ** (lengths[2 * i] + 1)),
                    r(10 ** lengths[2 * i + 1] - 1, 10 ** (lengths[2 * i + 1] + 1)))
                   for i in range(self.num_obj)]

        answers = [str((a + b)) for a, b in numbers]

        return numbers, answers

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return self.numbers[idx], self.answers[idx]
