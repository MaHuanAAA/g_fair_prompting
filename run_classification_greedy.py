import argparse

import numpy as np

from data_utils import load_dataset
from utils import *
import sys
from importlib import reload
from copy import deepcopy
import itertools

from bloom import load_bloom


def main(args):
    """
    Run experiment or load past results, print accuracy
    """
    tokenizer, model_bloom = load_bloom(args)
    default_params = {
        'conditioned_on_correct_classes': True,
        'subsample_test_set': args.subsample_test_set,
        'api_num_log_prob': args.api_num_log_prob,
        'approx': args.approx,
        'bs': args.bs,
        'key_id': args.key_id,
        'loop_type':args.loop_type,
    }

    # list of all experiment parameters to run
    all_params = []
    for model in args.models:
        for dataset in args.datasets:
            for num_shots in args.all_shots:
                for seed in range(args.num_seeds[0], args.num_seeds[1]):
                    p = deepcopy(default_params)
                    p['model'] = model
                    p['dataset'] = dataset
                    p['seed'] = seed
                    p['num_shots'] = num_shots
                    p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['loop_type']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)


    # query the model and save the responses
    if args.use_saved_results:
        load_results(all_params)
    else:
        save_results(all_params, tokenizer, model_bloom)


def loop_list(list_origin, saved_idx, current_idx):
    list_tmp = [deepcopy(list_origin[i]) for i in(saved_idx)]
    list_tmp.insert(0, deepcopy(list_origin[current_idx]))
    return list_tmp


def save_results(params_list, tokenizer, model_bloom, freeze_test_set=True):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    result_tree = dict()
    for param_index, params in enumerate(params_list):
        print("\nExperiment name:", params['expr_name'])
        if param_index == 0:
            save_dict = dict_init()
        ### load data
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset(params)
        params_check(params,tokenizer=tokenizer, model_bloom=model_bloom)

        ### sample test set
        if params['subsample_test_set']==0:
            test_sentences, test_labels = all_test_sentences, all_test_labels
            print(f"selecting full test set ({len(all_test_labels)} examples)")
        else:
            if freeze_test_set:
                np.random.seed(0)  # always use seed 0 result if freeze
            else:
                np.random.seed(params['seed'])
            test_sentences, test_labels = random_sampling(all_test_sentences, all_test_labels,
                                                          params['subsample_test_set'])
            print(f"selecting {len(test_labels)} subsample of test set")

        ### sample few-shot training examples
        np.random.seed(params['seed'])
        shot_batch = int(params['num_shots']/4)
        train_sentences, train_labels= [], []
        for i in range(shot_batch):
            # when add new demonstrations, the old examples need $\in$ new set for comparison
            train_sentences_tmp, train_labels_tmp = random_sampling(all_train_sentences, all_train_labels, 4)
            train_sentences += train_sentences_tmp
            train_labels += train_labels_tmp

        ### Evaluate the performance and save all results
        # obtaining model's response on test examples
        print(f"getting raw resp for {len(test_sentences)} test sentences")
        save_dict['example_sentences'].append(train_sentences)
        save_dict['example_labels'].append(train_labels)
        save_dict['test_labels'].append(test_labels)
        # get prob for each label
        content_free_inputs = ["N/A", "", "[MASK]"]
        all_loop = len(train_labels)
        fair_idx, fair_value = [], 0
        for loop in range(all_loop):
            all_label_probs_loops, p_cf_loops, p_cf_loops_nonorm, remain_idx = [], [], [], []
            for loop_app in range(all_loop):
                train_sentences_tmp, train_labels_tmp = deepcopy(train_sentences), deepcopy(train_labels)
                if loop_app in fair_idx:
                    continue
                remain_idx.append(loop_app)
                train_sentences_tmp = loop_list(train_sentences_tmp, fair_idx, loop_app)
                train_labels_tmp = loop_list(train_labels_tmp, fair_idx, loop_app)
                p_cf_loops.append(get_p_content_free(params, train_sentences_tmp, train_labels_tmp,
                                   content_free_inputs=content_free_inputs, tokenizer=tokenizer,
                                   model_bloom=model_bloom))
            max_fair_idx, max_fair_value = cal_fair(p_cf_loops)
            if max_fair_value < fair_value:
                break
            fair_idx.insert(0,remain_idx[max_fair_idx])
            fair_value = max_fair_value
        save_dict["index"].append(fair_idx)
        train_sentences_tmp = [train_sentences[i] for i in fair_idx]
        train_labels_tmp = [train_labels[i] for i in fair_idx]
        raw_resp_test = get_model_response(params, train_sentences_tmp, train_labels_tmp, test_sentences, tokenizer, model_bloom)
        all_label_probs = get_label_probs(params, raw_resp_test, train_sentences_tmp, train_labels_tmp, test_sentences,tokenizer=tokenizer, model_bloom=model_bloom)
        content_free_inputs = ["N/A", "", "[MASK]"]
        p_cf = get_p_content_free(params, train_sentences_tmp, train_labels_tmp, content_free_inputs=content_free_inputs,tokenizer=tokenizer, model_bloom=model_bloom)
        acc_original = eval_accuracy(all_label_probs, test_labels)
        acc_calibrated = eval_accuracy(all_label_probs, test_labels, mode="diagonal_W", p_cf=p_cf)
        print(acc_original, acc_calibrated)
        save_dict["ori_acc_select"].append(acc_original)
        save_dict["cal_acc_select"].append(acc_calibrated)
        if params['seed'] == args.num_seeds[-1] - 1:
            save_dict['settings'] = args
            SAVE_DIR = os.path.join(args.save_dir, 'bloom')
            if not os.path.isdir(SAVE_DIR):
                os.mkdir(SAVE_DIR)
                print(f"mkdir at {SAVE_DIR} for saving results")
            LOG_SAVE_DIR = os.path.join(SAVE_DIR, '{}_{}_{}shot_{}test_{}_seeds{}'.format(str(params['model']),
                                                                                          str(params['dataset']),
                                                                                          str(params['num_shots']),
                                                                                          str(params[
                                                                                                  'subsample_test_set']),
                                                                                          params['loop_type'],
                                                                                          str(args.num_seeds[
                                                                                                  0]) + 'to' + str(
                                                                                              args.num_seeds[-1])))
            np.save(LOG_SAVE_DIR, save_dict)
    print_results(result_tree)

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False
    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        # print('W',W)
        # print('b',b)
        # print('cp',calibrate_label_probs)
        ans_label = np.argmax(calibrate_label_probs)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)

def get_label_probs(params, raw_resp, train_sentences, train_labels, test_sentences, tokenizer, model_bloom):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(params['label_dict'])
    approx = params['approx']
    assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
        label_probs = [0] * len(params['label_dict'].keys())
        for j, label_list in params['label_dict'].items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # notice prompt does not have space after 'A:'
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j) # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs) # prob not normalized

    # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
    # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
    if (not approx) and (len(all_missing_positions) > 0):
        print(f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}")
        all_additional_prompts = []
        num_prompts_each = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            test_sentence = test_sentences[which_sentence]
            label_list = params['label_dict'][which_label]
            for label in label_list:
                prompt = construct_prompt(params, train_sentences, train_labels, test_sentence)
                prompt += " " + label
                all_additional_prompts.append(prompt)
            num_prompts_each.append(len(label_list))

        # chunk the prompts and feed into model
        chunked_prompts = list(chunks(all_additional_prompts, chunk_size_helper(params)))
        all_probs = []
        for chunk_id, chunk in enumerate(chunked_prompts):
            resp = complete(chunk, 0, tokenizer=tokenizer, model_bloom=model_bloom, echo=True, num_log_probs=1)
            for ans in resp['choices']:
                prob = np.exp(ans['logprobs']['token_logprobs'][-1])
                all_probs.append(prob)

        assert sum(num_prompts_each) == len(all_probs)
        assert len(num_prompts_each) == len(all_missing_positions)

        # fill in corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)
            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        assert (all_label_probs > 0).all(), "all should be populated with non-zero value"

    return all_label_probs # NOT NORMALIZED

def get_p_content_free(params, train_sentences, train_labels, tokenizer, model_bloom, content_free_inputs=('N/A',)):
    """Query model with content free input, return its prediction probability for each label"""
    label_dict = params['label_dict']

    all_p_y = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt(params, train_sentences, train_labels, content_free_input)
        # print(prompt)
        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += np.exp(complete(prompt + " " + a,
                                        0, tokenizer=tokenizer, model_bloom=model_bloom,
                                        echo=True,
                                        num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1])
            p_y[i] = prob
        all_p_y.append(p_y)
    print('p_y', all_p_y)
    p_y = np.mean(np.array(all_p_y), axis=0)
    print('py_mean', p_y)
    p_y_norm = p_y / np.sum(p_y) # normalize
    return p_y_norm

def params_check(params,tokenizer, model_bloom):
    """sanity check the experiment params"""
    assert params['num_tokens_to_predict'] == 1
    # for classification, make sure that all of the class names are one word.
    for key, label_names in params['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            first_token_of_label_name = complete(' ' + label_name, 1, tokenizer=tokenizer, model_bloom=model_bloom, echo=True, num_log_probs=2)['choices'][0]['logprobs']['tokens'][0]
            if first_token_of_label_name[1:] != label_name:
                print('label name is more than 1 token', label_name)
                assert False

    if not (params['dataset'] in ['rte']):
        # formatting: there should be a space after question/answer prefix
        assert params["q_prefix"][-1] == " "
        assert params["a_prefix"][-1] == " "
        assert len(params["prompt_prefix"]) == 0 or params["prompt_prefix"][-2:] == '\n\n'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True, help='num seeds for the training set')
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True, help='num training examples to use')
    parser.add_argument('--loop_type', dest='loop_type', action='store', required=True,
                        help='greedy or topk')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=3, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=1,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    parser.add_argument('--key_id', dest='key_id', action='store', required=False, type=str, default='0')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True, default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')
    parser.add_argument("--save_dir", type=str, help="dir to save the results")
    parser.add_argument("--model-dir", type=str, help="the path of pre-train model")
    parser.add_argument("--hidden-size", type=int, default=14336, help="hidden size")
    parser.add_argument("--max-length", type=int, default=2048, help="the max length of input")
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="int8")

    args = parser.parse_args()

    args.models = convert_to_list(args.models)
    args.datasets = convert_to_list(args.datasets)
    args.all_shots = convert_to_list(args.all_shots, is_int=True)
    args.num_seeds = convert_to_list(args.num_seeds, is_int=True)

    main(args)
