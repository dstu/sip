import os
import csv
import itertools
import numpy as np

from sip.data_gen.gen_isl import *
from sip.data_gen.isl_sampling_utilities import *
from sip.data_gen.utils import write_tsv

import transformers
from config_evaluator import Lazy
from sip.data_loading import *
from sip.embed_finetune import *
from sip.task_finetune import *

class NotEnoughExamplesError(Exception):
    def __init__(self, nn, balance):
        self.nn = nn
        self.balance = balance

    def __str__(self):
        if self.balance:
            return f"Cannot find {nn//2} examples of each class."
        else:
            return f"Cannot find {nn} examples."

def gen_phonology_problem(vocabulary, replace_fn, balance=True, n_exes=20):
    v_order = vocabulary[:]
    np.random.shuffle(v_order)

    e_changed = []
    e_unchanged = []
    e_all = []
    for vi in v_order:
        ri = replace_fn(vi)
        e_all.append((vi, ri))
        if ri != vi:
            e_changed.append((vi, ri))
        else:
            e_unchanged.append((vi, ri))

        if balance:
            if len(e_changed) >= n_exes // 2 and len(e_unchanged) >= n_exes // 2:
                break
        else:
            if len(e_all) >= n_exes:
                break

    if balance:
        if len(e_changed) >= n_exes // 2 and len(e_unchanged) >= n_exes // 2:
            return e_changed[:n_exes // 2], e_unchanged[:n_exes // 2]
    else:
        if len(e_all) >= n_exes:
            return e_all[:n_exes]

    raise NotEnoughExamplesError(n_exes, balance)

def train_test_split(changed, unchanged, n_test):
    test = changed[:n_test // 2] + unchanged[:n_test // 2]
    train = changed[n_test//2:] + unchanged[n_test//2:]
    np.random.shuffle(train)
    np.random.shuffle(test)
    return train, test

class ExperimentLogger:
    def __init__(self, writer, num_train, sample):
        self.writer = writer
        self.step = itertools.count()
        self.num_train = num_train
        self.sample = sample

    def progress_bar(self, generator):
        for item in generator:
            yield item

    def log_metrics(self, name, fields):
        if "loss" in fields:
            return #don't need this
        fields["sample"] = self.sample
        fields["step"] = next(self.step)
        fields["num_train"] = self.num_train
        self.writer.writerow(fields)

def run_experiment(words, process, run, num_train, num_test, n_trials=64):
    os.makedirs(f"data/eval/{run}", exist_ok=True)
    with open(f"data/eval/{run}/scores.tsv", "w") as sfh:
        fields = ["num_train", "sample", "step", "acc", "edit_dist", "per",
                  "acc_avg_10", "edit_dist_avg_10", "per_avg_10"]
        scoreWriter = csv.DictWriter(sfh, fieldnames=fields, dialect="excel-tab")
        
        for ii in range(n_trials):
            changed, unchanged = gen_phonology_problem(words, process, balance=True, n_exes=num_train + num_test)
            train, test = train_test_split(changed, unchanged, num_test)
            train = [(inp + chr(SYMBOL_RDELIM), outp) for (inp, outp) in train]
            test = [(inp + chr(SYMBOL_RDELIM), outp) for (inp, outp) in test]
            write_tsv(f"data/eval/{run}/i_{ii}_t_{num_train}_v_{num_test}_train.tsv", train)
            write_tsv(f"data/eval/{run}/i_{ii}_t_{num_train}_v_{num_test}_val.tsv", test)

            logger = ExperimentLogger(scoreWriter, num_train=num_train, sample=ii)
            
            #copied from sip_isl_canon.jsonnet
            tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")
            train_loader = prepare_task_dataset(batch_size=32,
                                                path=f"data/eval/{run}/i_{ii}_t_{num_train}_v_{num_test}_train.tsv",
                                                tokenizer=tokenizer)
            val_loader = prepare_task_dataset(batch_size=32,
                                                path=f"data/eval/{run}/i_{ii}_t_{num_train}_v_{num_test}_val.tsv",
                                                tokenizer=tokenizer)
            model = load_struct_prefix_with_init(
                prefix_length=50,
                num_examples=32,
                random_selection=True,
                fst_tokenizer_path="unicode_char_tokenizer_ipa.json",
                tokenizer=tokenizer,
                model_str="models/w_fsts_pretrain_s4_32",
                fst_format="isl_canon",
                map_location="cpu")
            
            finetune_model(model=model,
                           tokenizer=tokenizer,
                           train_data_loader=train_loader,
                           validation_data_loader=val_loader,
                           optimizer=Lazy({"lr":5e-4}, torch.optim.Adam),
                           num_epochs=50,
                           optimizer_groups=[
                               [".*prefix_embedding.*", {"lr": 1.0}],
                               [".*", {"lr": 3e-4}],
                           ],
                           grad_scale=1.0,
                           logger=logger,
                           )
            #score = evaluate_model(...)
            #scoreWriter.write_row({ "index" : ii, "score" : score})

if __name__ == "__main__":
    #implement a simple assimilation pattern
    toy_vocab = ["inner", "inmer", "inter"]
    toy_alphabet = set()
    for vi in toy_vocab:
        toy_alphabet.update(vi)
    toy_alphabet = "".join(toy_alphabet)
    print("Alphabet:", toy_alphabet)
        
    fst = make_2isl_transducer(
        [
            (("n", "m"), ("m", "m")),
        ],
        toy_alphabet)

    print_fst(fst)
    test_fst(fst, "inmi", "immi")

    fst = postprocess_for_sampling(fst)
    def run_fst(ss):
        return apply_fst(fst, ss)

    for ss in toy_vocab:
        print(ss, run_fst(ss))

    german = np.loadtxt("https://raw.githubusercontent.com/unimorph/deu/refs/heads/master/deu",
                        dtype=str, delimiter="\t")
    german_words = list([xx for xx in german[:, 1] if xx.isalpha()])
    german_chars = set()
    for vi in german_words:
        german_chars.update(vi)
    german_alphabet = "".join(sorted(list(german_chars)))
    print(german_alphabet)

    devoice_d = make_2isl_transducer(
        [
            (("d", "</s>"), ("t",)),
        ],
        german_alphabet)

    devoice_d = postprocess_for_sampling(devoice_d)
    def run_devoice(ss):
        return apply_fst(devoice_d, ss)

    change, unchange = gen_phonology_problem(german_words, run_devoice)
    for pair in change:
        print(pair)
    print()
    for pair in unchange:
        print(pair)
    print("-----")

    train, test = train_test_split(change, unchange, 8)
    for pair in train:
        print(pair)
    print()
    for pair in test:
        print(pair)
        
    run_experiment(german_words, run_devoice, "devoice_d", 10, 64)
