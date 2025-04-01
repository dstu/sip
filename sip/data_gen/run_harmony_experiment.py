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
            f"{trigger}{char}",
            f"{trigger}.{char}",
            f"{trigger}.*{char}",
            f"{char}"]
        for pattern in patterns:
            if re.search(re.compile(pattern), word):
                return pattern
        return None

    catWords = defaultdict(list)
    for word in spanish_words:
        catWords[categorize(word, "e", "a")].append(word)

    for cat, words in catWords.items():
        print(cat, ":", words[:10])

    def run_fn_s1(ss): #local progressive change
        return re.sub("ae", "aA", ss)

    def run_fn_s2(ss): #progressive change skipping a single position
        return re.sub("a(.)e", "a\\1A", ss)

    def run_fn_s3(ss): #progressive change skipping up to a single position
        return re.sub("a(.)?e", "a\\1A", ss)

    def run_fn_s4(ss): #progressive harmony
        sOld = ss
        sNew = re.sub("a(.*)e", "a\\1A", ss)
        while sNew != sOld:
            sOld = sNew
            sNew = re.sub("a(.*)e", "a\\1A", sOld)
        return sNew
    
    mode = int(sys.argv[1])
    assert(0 < mode <= 4)
    fns = [run_fn_s1, run_fn_s2, run_fn_s3, run_fn_s4]
    fn = fns[mode - 1]
    print("Running for mode", mode - 1)

    train, test = gen_balanced_problem(catWords, fn, 4, 2)
    for pair in train:
        print(pair)
    print()
    for pair in test:
        print(pair)
    print("-----")

    run_name = f"harmony_{mode}"
    os.makedirs(f"data/eval/{run_name}", exist_ok=True)
    with open(f"data/eval/{run_name}/scores.tsv", "w") as sfh:
        fields = ["num_train", "sample", "step", "acc", "edit_dist", "per",
                  "acc_avg_10", "edit_dist_avg_10", "per_avg_10"]
        scoreWriter = csv.DictWriter(sfh, fieldnames=fields, dialect="excel-tab")
        scoreWriter.writeheader()

    for size in range(2, 40, 2):
        run_experiment(catWords, fn, run_name, size, 8, n_trials=16)
