import torch
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast(vocab_file='../vocab/vocab.txt',
                              sep_token="[SEP]",
                              pad_token="[PAD]",
                              cls_token="[CLS]")

# def mock_inputs():
#     input_ids = torch.tensor([[101, 5401, 1957, 5276, 1658,  102, 2458, 1962, 2791, 5023,  872,  749,
#                                102, 2769, 3341, 1568,  102,    0,    0,    0,    0,    0,    0,    0,
#                                0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
#                               [101, 2682, 4692,  872, 4638, 5401, 4212,  102,  779, 2769,  671, 1366,
#                                2218, 5314,  872, 4692,  102, 2769,  779,  697, 1366,  102, 6374, 1328,
#                                782, 2157, 2897, 2207, 2891, 2891, 2948,  872, 5541, 1366,  102]], dtype=torch.int64)

#     labels = torch.tensor([[101, 5401, 1957, 5276, 1658,  102, 2458, 1962, 2791, 5023,  872,  749,
#                             102, 2769, 3341, 1568,  102, -100, -100, -100, -100, -100, -100, -100,
#                             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
#                            [101, 2682, 4692,  872, 4638, 5401, 4212,  102,  779, 2769,  671, 1366,
#                             2218, 5314,  872, 4692,  102, 2769,  779,  697, 1366,  102, 6374, 1328,
#                             782, 2157, 2897, 2207, 2891, 2891, 2948,  872, 5541, 1366,  102]], dtype=torch.int64)

#     return input_ids, labels


def mock_inputs():
    txt = '你好'
    input_ids = tokenizer.encode(txt, add_special_tokens=True)
    input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
    return input_ids, input_ids
    ...


def mock_hidden_states(config):
    batch_size = 2
    seq_len = 35
    n_embd = config.n_embd
    hidden_states = torch.randn(size=(batch_size, seq_len, n_embd))
    return hidden_states


if __name__ == "__main__":
    mock_inputs()
    ...