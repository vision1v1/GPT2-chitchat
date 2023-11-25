import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention, GPT2MLP, Conv1D
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.activations import NewGELUActivation
from mock_data import mock_inputs, mock_hidden_states

# model_config_path = '../config/config.json'
model_config_path = '../config/debug_config.json'
model_config = GPT2Config.from_json_file(model_config_path)
# print("config =", model_config, sep='\n', end='\n\n')


def debug_gpt2_lmhead_model():
    model = GPT2LMHeadModel(config=model_config)
    print("model = ", model, sep='\n', end='\n\n')
    input_ids, labels = mock_inputs()
    output:CausalLMOutputWithCrossAttentions = model.forward(input_ids=input_ids, labels=labels)
    loss = output.loss # input_ids 与 labels 之间的损失。
    logists = output.logits # softmax 之前的预测
    output.attentions # 所有block的 self attention
    output.cross_attentions # 所有block的 cross_attention
    output.past_key_values # 

    print("output.loss = ", loss, sep='\n', end='\n\n')
    print("output.logits = ", logists, sep='\n', end='\n\n')


def debug_gpt2_model():
    model = GPT2Model(config=model_config)
    print("model = ", model, sep='\n', end='\n\n')
    input_ids, labels = mock_inputs()
    output:BaseModelOutputWithCrossAttentions = model.forward(input_ids=input_ids, return_dict=True)
    last_hidden_state = output.last_hidden_state # 最后一个block的输出
    hidden_states = output.hidden_states # 所有block的输出，需要设置参数，才输出，默认不输出
    attentions = output.attentions # 所有自注意的attention，需要设置参数，才输出，默认不输出
    cross_attentions = output.cross_attentions # 所有的cross_attention, 需要设置参数，才输出，默认不输出
    print("output.last_hidden_state = ", last_hidden_state, sep='\n', end='\n\n') 
    print("output.hidden_states = ", hidden_states, sep='\n', end='\n\n')
    print("output.attentions = ", attentions, sep='\n', end='\n\n')
    print("output.cross_attentions = ", cross_attentions, sep='\n', end='\n\n')



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
