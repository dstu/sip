import torch

import Levenshtein

from collections import deque, Counter

from sip.data_gen.utils import write_tsv

class MovingAvg:
    # Naive implementation
    def __init__(self, window_length):
        self.l = deque()
        self.window_length = window_length

    def extend(self, l):
        last = None
        for x in l:
            last = self.append(x)
        return last

    def get(self):
        if len(self.l) == 0:
            return 0
        return sum(self.l) / len(self.l)

    def append(self, x):
        if len(self.l) == self.window_length:
            self.l.popleft()
        self.l.append(x)
        return sum(self.l) / len(self.l)

def add_binary_stats(stats, results):
    for (inp, gold, pred) in results:
        inp = inp.rstrip(u"\x13")
        if inp == gold:
            trial_type = "neg"
        else:
            trial_type = "pos"

        if inp == pred:
            resp_type = "neg"
        elif gold == pred:
            resp_type = "corr"
        else:
            resp_type = "err"

        #print("Trial:", inp, gold, pred, trial_type, resp_type)
        stats[(trial_type, resp_type)] += 1

def binary_metrics(stats):
    nn = sum(stats.values())

    totalPos = 0
    totalNeg = 0
    for (ttp, rtp), ct in stats.items():
        if ttp == "pos":
            totalPos += ct
        else:
            totalNeg += ct

    #print("total", nn, totalPos, totalNeg)
    assert(totalPos + totalNeg == nn)

    tpr = stats[("pos", "corr")] / totalPos
    fpr = stats[("neg", "err")] / totalNeg
    tnr = stats[("neg", "neg")] / totalNeg
    ifn = tpr + tnr - 1
    return { "tpr" : tpr, "fpr" : fpr, "tnr" : tnr, "inform" : ifn }

def evaluate_on(model, tokenizer, dataloader, log_result=None):
  correct, total, edit_dist, per = 0,0,0,0
  binary_stats = Counter()
  model.eval()
  all_preds = []
  for test_batch in dataloader:
    test_batch = {k: v.to(model.device) for k,v in test_batch.items()}
    test_batch_inputs = dict(test_batch)
    inp = tokenizer.batch_decode(100*(test_batch_inputs["input_ids"] == -100) + test_batch_inputs["input_ids"], skip_special_tokens=True) # replace -100 by 0
    del test_batch_inputs["labels"]
    r = tokenizer.batch_decode(model.generate(**test_batch_inputs, max_new_tokens=test_batch["labels"].shape[1]+2,
                                              early_stopping="never", num_beams=1, no_repeat_ngram_size=0), skip_special_tokens=True)
    gold = tokenizer.batch_decode(100*(test_batch["labels"] == -100) + test_batch["labels"], skip_special_tokens=True) # replace -100 by 0
    add_binary_stats(binary_stats, zip(inp, gold, r))

    if log_result != None:
        all_preds += list(zip(gold, r))

    correct += sum( [x == y for x,y in zip(r, gold)])
    total += len(gold)
    edit_dist += sum( Levenshtein.distance(x,y) for x,y in zip(r, gold))
    per += sum(Levenshtein.distance(x,y)/max(1, len(y)) for x,y in zip(r, gold))

  if log_result != None:
      write_tsv(log_result, all_preds)
  return correct/total, edit_dist/total, per/total, binary_metrics(binary_stats)

#################
from math import ceil
def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(devices, layers_list))
    
def hack_t5_parallelize(model):
   model.encoder.parallelize(get_device_map(len(model.encoder.block), range(torch.cuda.device_count())))
   model.decoder.parallelize(get_device_map(len(model.decoder.block), range(torch.cuda.device_count())))
   model.lm_head = model.lm_head.to(model.decoder.first_device)
   model.model_parallel = True

   return model

