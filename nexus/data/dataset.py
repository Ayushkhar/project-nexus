import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


def format_input(entry):
    """
    Converts instruction dataset entries into Alpaca-style prompt format.
    """

    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class GPTDatasetV1(Dataset):
    """
    Dataset for training GPT models using sliding window token sequences.
    """

    def __init__(self, txt, tokenizer, max_length, stride):

        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):

            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
        txt,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0
):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


class InstructionDataset(Dataset):
    """
    Dataset used for instruction tuning (Alpaca-style).
    """

    def __init__(self, data, tokenizer):

        self.data = data
        self.encoded_texts = []

        for entry in data:

            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"

            full_text = instruction_plus_input + response_text

            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return torch.tensor(self.encoded_texts[index])

    def __len__(self):
        return len(self.data)