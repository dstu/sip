import dataclasses
import json
import math
import os
import sys
import ast
from typing import List, Tuple, Set, Iterable

import random

import numpy as np

import pynini, pywrapfst

import time
import tqdm

from sip.data_gen.gen_isl import select_factors, make_2isl_transducer, NotKISLError, replace_star_transitions, replace_star_state, print_fst, SYMBOL_RDELIM
from sip.data_gen.isl_sampling_utilities import postprocess_for_sampling, gen_and_recode_pair, recode_string, fst_to_json
from sip.data_gen.utils import gen_pair, one_step, FSTCollection, random_subset, replace_arc

# use some ASCII control codes to take special meaning.
SYMBOL_ID = 17
SYMBOL_TO_UPPER = 18
SYMBOL_TO_LOWER = 19

vocab = [chr(x) for x in range(32, 127)]
vocab = vocab + [chr(i) for i in range(592, 687+1)] # add unicode characters for IPA symbols.
vocab = sorted(set(vocab))
#these characters have special meaning in OpenFST, and cannot be written into FAR
vocab.remove("]")
vocab.remove("[")
vocab.remove(chr(92)) # backslash, this messes things up as well!
vocab.remove("*") #used as internal representation for "any char" within the isl code
vocab.remove("(") #so we can assume this character is always the first of a portmanteau string

if __name__ == "__main__":
    os.makedirs("data/pretrain_2isl", exist_ok=True)
    t0 = time.time()
    random.seed(55)

    print(vocab)

    fst_collection = FSTCollection()
    num_data_points = 50_000
    # num_data_points = 10_000
    # num_data_points = 10
    num_fsts = 5*num_data_points # make sure we don't run out of seeds
    num_ex_per_task = 5
    seeds = [random.randint(0, 100000000000) for _ in range(num_fsts)]

    DESIRED_MAX_FACTORS = 4
    # edit this line to set the representation to "isl" or "canonical"
    REPRESENTATION = "canonical"

    name = f"pretrain_s{DESIRED_MAX_FACTORS}_{REPRESENTATION}"

    max_num_factors = 0

    for seed in tqdm.tqdm(seeds):
        num_factors = random.randint(1, DESIRED_MAX_FACTORS)
        vocab_size = random.randint(5, 25)
        my_vocab = list(vocab)
        random.shuffle(my_vocab)
        chosen_vocab = "".join(my_vocab[:vocab_size])
        factors = select_factors(num_factors, 2, chosen_vocab, p_progressive=0.25, p_regressive=0.25,
                                 epsilon_allowed=True, p_epenthesis=0.05)
        try:
            fst = make_2isl_transducer(factors, chosen_vocab, minimize=(REPRESENTATION == "canonical"))
            print("Factors:", factors)
            # print_fst(fst)
            # their code has a validity check here to make sure none of the arcs
            # have invalid character codes in the sampling machine
            # we don't need that, but it is possible to get extra-long output symbols
            fst_invalid = False
            osyms = fst.fst.output_symbols()
            for ii in range(osyms.num_symbols()):
                key = osyms.get_nth_key(ii)
                val = osyms.find(key)
                if val.startswith("("):
                    val = ast.literal_eval(val)
                    if len(val) > 2:
                        fst_invalid = True

            if fst_invalid:
                print("Continuing: fst is not valid")
                continue
        except NotKISLError:
            print("Continuing: fst is not k-isl")
            continue

        max_num_factors = max(max_num_factors, len(factors))

        # all 2ISL languages are cyclic except the empty language
        # because the factor # a # is 3 chars long, so it's not possible to see both boundaries at once
        # so we can omit the cyclicity check
        fst_collection.maybe_add(fst, chosen_vocab)

        print("check on length:", len(fst_collection), len(fst_collection.to_list()))
        assert(len(fst_collection) == len(fst_collection.to_list()))

        if len(fst_collection) > num_data_points:
            print("fst collection is large enough to halt", len(fst_collection), num_data_points)
            break

    fst_collection = fst_collection.to_list()
    random.shuffle(fst_collection)

    if len(fst_collection) < num_data_points:
        print("Collected", len(fst_collection), "items but wanted", num_data_points)
        raise ValueError("fst collection not large enough")

    # split into train/dev/test

    collection_ids = list(range(min(num_data_points, len(fst_collection))))
    random.shuffle(collection_ids)

    num_train_ex = int(0.8 * len(collection_ids))
    num_easy_dev_ex = min(1000, num_train_ex)
    num_dev_ex = min(1000, int(0.1 * len(collection_ids)))
    num_test_ex = min(1000, int(0.1 * len(collection_ids)))

    curr_train = 0
    curr_dev = 0
    curr_test = 0
    curr_easy_dev = 0

    max_length_json = 0
    task_id = 0

    max_digits = len(str(len(fst_collection)))
    with (open(f"data/pretrain_2isl/train_{name}.jsonl", "w") as f_train,
          pynini.Far(f"data/pretrain_2isl/train_{name}.far", mode="w") as far_train,
          pynini.Far(f"data/pretrain_2isl/dev_{name}.far", mode="w") as far_dev,
          pynini.Far(f"data/pretrain_2isl/test_{name}.far", mode="w") as far_test,
          open(f"data/pretrain_2isl/dev_{name}.jsonl", "w") as f_dev,
          open(f"data/pretrain_2isl/easy_dev_{name}.jsonl", "w") as easy_dev_f,
          open(f"data/pretrain_2isl/test_{name}.jsonl", "w") as f_test):

        for fst, chosen_vocab in tqdm.tqdm(fst_collection):

            fst_for_sampling = postprocess_for_sampling(fst)
            length_restriction = one_step(fst_for_sampling.fst.input_symbols()).closure(1, 35)
            delimited_length_restriction = (length_restriction +
                                            pynini.accep("</s>", token_type=fst_for_sampling.fst.input_symbols()))
            train_fst = pynini.compose(delimited_length_restriction, fst_for_sampling.fst)
            
            if train_fst.num_states() == 0:
                # Occasionally, this might happen, e.g. if the have a LOWER operation but no characters can be converted to lowercase (vocab is all symbols)
                # and this transition is the only way to get to a final state.
                continue

            task_id += 1

            fst_as_json = fst_to_json(fst)
            max_length_json = max(max_length_json, len(fst_as_json))
            data_points = []
            for _ in range(num_ex_per_task):
                inp, o = gen_and_recode_pair(train_fst,
                                             fst_for_sampling.fst.input_symbols(),
                                             fst_for_sampling.fst.output_symbols(),
                                             seed=random.randint(0, 100000000000))
                #print(f"i!{inp}!i o!{o}!o")
                # assert pynini.compose(pynini.accep(inp, token_type="utf8"), train_fst).num_states() > 0
                # assert pynini.compose(train_fst, pynini.accep(o, token_type="utf8")).num_states() > 0

                #checks that the i/o pair is validly produced by the transducer
                #but vanilla 'accep' will not work for custom symtabs
                #assert pynini.compose(pynini.compose(pynini.accep(inp, token_type="utf8"), train_fst), pynini.accep(o, token_type="utf8")).num_states() > 0

                data_points.append({"FST": fst_as_json, "input": inp, "output": o, "task_id": task_id})

            task_id_s = str(task_id)
            task_id_s = "0" * (max_digits - len(task_id_s)) + task_id_s

            if curr_train < num_train_ex:
                f = f_train
                curr_train += 1
                far = far_train
            elif curr_dev < num_dev_ex:
                f = f_dev
                curr_dev += 1
                far = far_dev
            elif curr_test < num_test_ex:
                f = f_test
                curr_test += 1
                far = far_test
            else:
                break

            far.add(task_id_s, fst.fst)

            for datapoint in data_points:
                f.write(json.dumps(datapoint))
                f.write("\n")

            #If we are still generating training data, generate some easy dev examples (= known tasks but unkown strings) as well
            if curr_train <= num_train_ex and curr_easy_dev < num_easy_dev_ex:
                curr_easy_dev += 1
                inputs = set(datapoint["input"] for datapoint in data_points)
                excluded_inputs = pynini.union(*[recode_string(input, fst_for_sampling.fst.input_symbols())
                                                 for input in inputs])

                sigma_star = one_step(fst_for_sampling.fst.input_symbols()).closure()

                allowed_inputs = pynini.difference(sigma_star, excluded_inputs)
                easy_dev_fst = pynini.compose(allowed_inputs, train_fst)

                for _ in range(num_ex_per_task):
                    inp, o = gen_and_recode_pair(easy_dev_fst,
                                                 fst_for_sampling.fst.input_symbols(),
                                                 fst_for_sampling.fst.output_symbols(),
                                                 seed=random.randint(0, 100000000000))
                    #print(f"I!{inp}!I O!{o}!O")
                    assert inp not in inputs
                    #assert pynini.compose(pynini.accep(inp, token_type="utf8"), train_fst).num_states() > 0

                    easy_dev_f.write(json.dumps({"FST": fst_as_json, "input": inp, "output": o, "task_id": task_id}))
                    easy_dev_f.write("\n")

    print("Max num. factors", max_num_factors)
    print("Max num. transitions", max_length_json)
