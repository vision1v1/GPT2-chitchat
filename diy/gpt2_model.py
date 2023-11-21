import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP
from transformers.activations import NewGELUActivation

model_config_path = '../config/config.json'
model_config = GPT2Config.from_json_file(model_config_path)
print("config =", model_config, sep='\n', end='\n\n')


def mock_inputs():
    input_ids = torch.tensor([[101, 5401, 1957, 5276, 1658,  102, 2458, 1962, 2791, 5023,  872,  749,
                               102, 2769, 3341, 1568,  102,    0,    0,    0,    0,    0,    0,    0,
                               0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],
                              [101, 2682, 4692,  872, 4638, 5401, 4212,  102,  779, 2769,  671, 1366,
                               2218, 5314,  872, 4692,  102, 2769,  779,  697, 1366,  102, 6374, 1328,
                               782, 2157, 2897, 2207, 2891, 2891, 2948,  872, 5541, 1366,  102]], dtype=torch.int64)

    labels = torch.tensor([[101, 5401, 1957, 5276, 1658,  102, 2458, 1962, 2791, 5023,  872,  749,
                            102, 2769, 3341, 1568,  102, -100, -100, -100, -100, -100, -100, -100,
                            -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
                           [101, 2682, 4692,  872, 4638, 5401, 4212,  102,  779, 2769,  671, 1366,
                            2218, 5314,  872, 4692,  102, 2769,  779,  697, 1366,  102, 6374, 1328,
                            782, 2157, 2897, 2207, 2891, 2891, 2948,  872, 5541, 1366,  102]], dtype=torch.int64)
    
    return input_ids, labels


def debug_gpt2_lmhead_model():
    model = GPT2LMHeadModel(config=model_config)
    print("model = ", model, sep='\n', end='\n\n')
    input_ids, labels = mock_inputs()
    output = model.forward(input_ids=input_ids, labels=labels)
    print("output = ", output, sep='\n', end='\n\n')


def debug_gpt2_model():
    model = GPT2Model(config=model_config)
    print("model = ", model, sep='\n', end='\n\n')
    input_ids, labels = mock_inputs()
    output = model.forward(input_ids=input_ids, return_dict=True)
    print("output = ", output, sep='\n', end='\n\n')


def debug_gpt2_block():
    block = GPT2Block(config=model_config, layer_idx=0)
    print("block = ", block, sep='\n', end='\n\n')
    batch_size = 2
    seq_len = 35
    n_embd = model_config.n_embd
    hidden_states = torch.randn(size=(batch_size, seq_len, n_embd))
    output = block.forward(hidden_states=hidden_states)
    print("output = ", output, sep='\n', end='\n\n')
    ...


def debug_gpt2_attention():
    attn = GPT2Attention(config=model_config)
    
    ...

if __name__ == "__main__":
    # debug_gpt2_lmhead_model()
    # debug_gpt2_model()
    debug_gpt2_block()
