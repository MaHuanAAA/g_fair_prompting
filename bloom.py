import os
import argparse
import math
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = torch.cuda.device_count()
rank = local_rank

def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)

print_rank0(f"Using {world_size} gpus")

def parse_args():
    parser = argparse.ArgumentParser(description="Bloom")

    parser.add_argument("--input", type=str, help="test data")
    parser.add_argument("--output", type=str, help="output data")
    parser.add_argument("--model-dir", type=str, help="the path of pre-train model")
    parser.add_argument("--batch-size", type=int, default=2, help="batch size")
    parser.add_argument("--hidden-size", type=int, default=14336, help="hidden size")
    parser.add_argument("--max-length", type=int, default=2048, help="the max length of input")
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="int8")

    return parser.parse_args()

def get_max_memory_per_gpu_dict(dtype, model_name):
    """try to generate the memory map based on what we know about the model and the available hardware"""

    # figure out the memory map - the minimum per gpu required to load the model
    n_gpus = torch.cuda.device_count()

    if (
        n_gpus == 8
        and torch.cuda.get_device_properties(0).total_memory > 79 * 2**30
    ):
        # hand crafted optimized memory map for 8x80 setup over BLOOM
        # this works with bs=40
        if dtype != torch.int8:
            max_memory_per_gpu = {
                0: "0GIB",
                1: "51GIB",
                2: "51GIB",
                3: "51GIB",
                4: "51GIB",
                5: "51GIB",
                6: "51GIB",
                7: "51GIB",
            }
        else:
            max_memory_per_gpu = {
                0: "0GIB",
                1: "26GIB",
                2: "26GIB",
                3: "26GIB",
                4: "26GIB",
                5: "26GIB",
                6: "26GIB",
                7: "26GIB",
            }
        print("Max memory per gpu:", max_memory_per_gpu)
        return max_memory_per_gpu

    try:
        # model_params calculation, as we don't have a model yet to do:
        # model_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

        config = AutoConfig.from_pretrained(model_name)
        h = config.hidden_size
        l = config.n_layer
        v = config.vocab_size
        # from https://github.com/bigscience-workshop/bigscience/tree/6917a3b5fefcf439d3485ca184b4d9f6ab605150/math#model-sizing
        model_params = l * (12 * h**2 + 13 * h) + v * h + 4 * h
    except:
        print_rank0(f"The model {model_name} has a broken config file. Please notify the owner")
        raise

    if dtype == torch.int8:
        bytes = 1
    else:
        bytes = torch.finfo(dtype).bits / 8
    param_memory_total_in_bytes = model_params * bytes
    # add 5% since weight sizes aren't the same and some GPU may need more memory
    param_memory_per_gpu_in_bytes = int(param_memory_total_in_bytes / n_gpus * 1.10)
    print_rank0(f"Estimating {param_memory_per_gpu_in_bytes/2**30:0.2f}GB per gpu for weights")

    # check the real available memory
    # load cuda kernels first and only measure the real free memory after loading (shorter by ~2GB)
    torch.ones(1).cuda()
    max_memory_per_gpu_in_bytes = torch.cuda.mem_get_info(0)[0]
    if max_memory_per_gpu_in_bytes < param_memory_per_gpu_in_bytes:
        raise ValueError(
            f"Unable to generate the memory map automatically as the needed estimated memory per gpu ({param_memory_per_gpu_in_bytes/2**30:0.2f}GB) is bigger than the available per gpu memory ({max_memory_per_gpu_in_bytes/2**30:0.2f}GB)"
        )

    max_memory_per_gpu = {i: param_memory_per_gpu_in_bytes for i in range(torch.cuda.device_count())}
    print("Max memory per gpu:", max_memory_per_gpu)
    return max_memory_per_gpu

def generate(inputs, tokenizer, model):
    """returns a list of zipped inputs, outputs and number of new tokens"""

    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    # get the max length for new tokens
    input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
    max_new_length=max(input_tokens_lengths)
    generate_kwargs = dict(max_new_tokens=min(max_new_length, 2048-max_new_length), do_sample=False)

    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(DEVICE)

    outputs = model.forward(input_ids=inputs)
    output_tokens_lengths = [x.shape[0] for x in outputs]

    total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return zip(inputs, outputs, total_new_tokens)

def load_bloom(args):
    # load pre-trained model
    model_name = args.model_dir
    print_rank0(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # XXX: can't automatically derive dtype via config's `from_pretrained`
    dtype = torch.bfloat16
    infer_dtype = args.dtype
    if infer_dtype == "int8":
        dtype = torch.int8

    kwargs = dict(
        device_map="auto",
        max_memory=get_max_memory_per_gpu_dict(dtype, model_name),
    )

    # kwargs = dict(
    #     device_map="balanced_low_0",
    # )

    if infer_dtype == "int8":
        print_rank0("Using `load_in_8bit=True` to use quanitized model")
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    print(f"{'=' * 60}")
    print("Model Bloom (int-8) is Successfully Loaded!")
    print(f"{'=' * 60}\n")
    return tokenizer, model

def inference(prompt, tokenizer, model, l=10, num_log_probs=None, echo=False):
    if isinstance(prompt, str):
        prompt = [prompt]  # the code below assumes a list
    input_ids = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    input_tokens_lengths = [x.shape[0] for x in input_ids.input_ids]
    max_new_length = max(input_tokens_lengths)
    generate_kwargs = dict(max_new_tokens=l, do_sample=False)

    for t in input_ids:
        if torch.is_tensor(input_ids[t]):
            input_ids[t] = input_ids[t].to(DEVICE)
    if l>0:
        total_sequences = model.generate(**input_ids, **generate_kwargs)
    else:
        assert echo == True and l == 0
        total_sequences = input_ids['input_ids'].cuda()
    # they want the probs of the top tokens
    if num_log_probs is not None:
        logits = model(total_sequences)['logits'].detach().to(dtype=torch.float32).cpu()
        if not echo:
            # get the top tokens and probs for the generated l tokens
            probs = torch.softmax(logits[:, -l - 1:], dim=2).cpu()
        else:
            # get the top tokens and probs for the context and the generated l tokens
            probs = torch.softmax(logits, dim=2).cpu()
        top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
        logprobs = torch.log(probs)
        top_log_probs = torch.log(top_probs)

    # create the return value to resemble OpenAI
    return_json = {}
    choices = []
    for batch_id in range(len(prompt)):
        curr_json = {}
        # text is just the optional context and next l tokens
        if not echo:
            curr_json['text'] = tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
        else:
            curr_json['text'] = tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

        # fill the return json with the top tokens and probs to match the OpenAI return value.
        if num_log_probs is not None:
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            if not echo:
                # cutoff the -1 here because the probs are shifted one over for LMs
                for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][:-1],
                                                                                     top_tokens[batch_id][:-1]):
                    # tokens is a list of the top token at each position
                    curr_json['logprobs']['tokens'].append(tokenizer.decode([current_element_top_tokens[0]]))
                    # token_logprobs is a list of the logprob of the top token at each position
                    curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                    # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
            else:
                # same as not above but small tweaks
                # we add null to the front because for the GPT models, they have null probability for the first token
                # (for some reason they don't have an beginning of sentence token)
                curr_json['logprobs']['top_logprobs'].append('null')
                # cutoff the -1 here because the probs are shifted one over for LMs
                for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(
                        zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])):
                    # skip padding tokens
                    if total_sequences[batch_id][index].item() == 50256:
                        continue
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
                for index in range(len(probs[batch_id])):
                    curr_json['logprobs']['tokens'].append(tokenizer.decode([total_sequences[batch_id][index]]))
                curr_json['logprobs']['token_logprobs'].append('null')
                for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
                    # probs are left shifted for LMs 
                    curr_json['logprobs']['token_logprobs'].append(
                        log_probs_token_position_j[total_sequences[batch_id][index + 1]])

        choices.append(curr_json)
    return_json['choices'] = choices
    return return_json



if __name__ == '__main__':
    tokenizer, model = load_bloom(parse_args())
    sentence = ["Obama was born in ", "today is a "]
    inference(sentence, tokenizer, model=model, num_log_probs=20)