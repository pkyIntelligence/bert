import torch


def create_padding_mask(seq):
    seq = torch.eq(seq, 0).float()

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    return torch.triu(torch.ones(size, size), diagonal=1)  # (size, size)
