import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP, Conv1D
from transformers.activations import NewGELUActivation
from mock_data import mock_inputs, mock_hidden_states

model_config_path = '../config/config.json'
model_config = GPT2Config.from_json_file(model_config_path)
print("config =", model_config, sep='\n', end='\n\n')


def debug_gpt2_lmhead_model():
    model = GPT2LMHeadModel(config=model_config)
    print("model = ", model, sep='\n', end='\n\n')
    input_ids, labels = mock_inputs()
    output = model.forward(input_ids=input_ids, labels=labels)
    print("output = ", output, sep='\n', end='\n\n')
    print("logists = ", output.logits, sep='\n', end='\n\n')


def debug_gpt2_model():
    model = GPT2Model(config=model_config)
    print("model = ", model, sep='\n', end='\n\n')
    input_ids, labels = mock_inputs()
    output = model.forward(input_ids=input_ids, return_dict=True)
    print("output = ", output, sep='\n', end='\n\n')


def debug_gpt2_block():
    block = GPT2Block(config=model_config, layer_idx=0)
    print("block = ", block, sep='\n', end='\n\n')
    hidden_states = mock_hidden_states(model_config)
    output = block.forward(hidden_states=hidden_states)
    print("output = ", output, sep='\n', end='\n\n')


def debug_gpt2_attention():
    attn = GPT2Attention(config=model_config)
    print("attn = ", attn, sep='\n', end='\n\n')
    hidden_states = mock_hidden_states(model_config)
    output = attn.forward(hidden_states=hidden_states)
    print("output = ", output, sep='\n', end='\n\n')
    ...


def debug_gpt2_mlp():
    mlp = GPT2MLP(intermediate_size=4 * model_config.hidden_size, config=model_config)
    print("mlp = ", mlp, sep='\n', end='\n\n')
    hidden_states = mock_hidden_states(model_config)
    output = mlp.forward(hidden_states=hidden_states)
    print("output = ", output, sep='\n', end='\n\n')


def debug_conv1d():
    embed_dim = model_config.n_embd
    conv1d = Conv1D(3 * embed_dim, embed_dim)
    print("conv1d = ", conv1d, sep='\n', end='\n\n')
    hidden_states = mock_hidden_states(model_config)
    output = conv1d.forward(hidden_states)
    print("output = ", output, sep='\n', end='\n\n')
    ...


def debug_new_gelue_activation():
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    act = NewGELUActivation()
    hidden_states = mock_hidden_states(model_config)
    output = act.forward(input=hidden_states)
    print("output = ", output, sep='\n', end='\n\n')
    ...


if __name__ == "__main__":
    debug_gpt2_lmhead_model()
    # debug_gpt2_model()
    # debug_gpt2_block()
    # debug_gpt2_attention()
    # debug_gpt2_mlp()
    # debug_conv1d()
    # debug_new_gelue_activation()
