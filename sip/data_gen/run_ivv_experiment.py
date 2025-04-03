import os
import csv
import re
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

from sip.data_gen.run_simple_experiment import *

if __name__ == "__main__":
    spanish = np.loadtxt("https://raw.githubusercontent.com/unimorph/spa/refs/heads/master/spa",
                        dtype=str, delimiter="\t")
    spanish_words = list([xx for xx in spanish[:, 1] if xx.isalpha()])
    spanish_chars = set()
    for vi in spanish_words:
        spanish_chars.update(vi)
    spanish_alphabet = "".join(sorted(list(spanish_chars)))
    print(spanish_alphabet)

    def categorize(word, char, trigger):
        patterns = [
            f"{trigger}{char}{trigger}",
            f"{char}{trigger}",
            f"{trigger}{char}",
            f"{char}"]
        for pattern in patterns:
            if pattern in word:
                return pattern
        return None

    catWords = defaultdict(list)
    for word in spanish_words:
        catWords[categorize(word, "d", "a")].append(word)
    
    def run_ivv(ss):
        vow = "AEIOUYaeiouyÉáâçéíîñóöúü"
        repl = {
            "b" : "B",
            "d" : "D",
            "g" : "G",
            }
        target = "".join(repl.keys())
        regexp = f"({[vow]})({[target]})({[vow]})"
        #print(regexp)
        def replacer(match):
            cons = repl.get(match.group(2), match.group(2))
            return match.group(1) + cons + match.group(3)
        result = re.sub(regexp, replacer, ss)
        return result

    def run_ivv_s1(ss):
        result = ss.replace("da", "Da")
        return result
    
    def run_ivv_s2(ss):
        result = ss.replace("da", "DDa")
        return result

    def run_ivv_s3(ss):
        result = ss.replace("ad", "aD")
        return result

    def run_ivv_s4(ss):
        result = ss.replace("ad", "aDD")
        return result
 
    def run_ivv_s5(ss):
        result = ss.replace("ada", "aDa")
        return result

    def run_ivv_s6(ss):
        result = ss.replace("ada", "aDDa")
        return result
    
    mode = int(sys.argv[1])
    assert(0 < mode <= 6)
    fns = [run_ivv_s1, run_ivv_s2, run_ivv_s3, run_ivv_s4, run_ivv_s5, run_ivv_s6]
    fn = fns[mode - 1]
    print("Running for mode", mode - 1)
    
    train, test = gen_balanced_problem(catWords, fn, 4, 2)
    for pair in train:
        print(pair)
    print()
    for pair in test:
        print(pair)
    print("-----")

    run_name = f"ivv_{mode}"
    os.makedirs(f"data/eval/{run_name}", exist_ok=True)
    with open(f"data/eval/{run_name}/scores.tsv", "w") as sfh:
        fields = ["num_train", "sample", "step", "acc", "edit_dist", "per",
                  "acc_avg_10", "edit_dist_avg_10", "per_avg_10"]
        scoreWriter = csv.DictWriter(sfh, fieldnames=fields, dialect="excel-tab")
        scoreWriter.writeheader()

    for size in range(2, 34, 2):
        run_experiment(catWords, fn, run_name, size, 8, n_trials=16)
