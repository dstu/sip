import sys
import re
import os
import glob
import csv
import numpy as np
from collections import *
import matplotlib.pyplot as plt

def plotRecs(ff, ax, title, yt, stat, xticks=False):
    with open(ff, "r") as ifh:
        reader = csv.DictReader(ifh, dialect="excel-tab")
        records = list(reader)

    #collate by training size
    bySize = defaultdict(list)
    for record in records:
        if int(record["num_train"]) > 28:
            continue
        bySize[int(record["num_train"])].append(record)

    seqs = []
    for ii, (size, records) in enumerate(sorted(bySize.items(), key=lambda xx: xx[0])):
        records = [float(record[stat]) for record in records if (int(record["step"]) == 49 and int(record["num_train"]) <=28)]
        #print(size, len(records))
        seqs.append(records)
        ax.text(ii - .25, .25 + (-1 * ii % 2) * .1, f"{np.median(records):.2}")

    ax.boxplot(seqs)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_ylabel(yt)
    if not xticks:
        ax.set_xticks([])
    else:
        ax.set_xticks(ax.get_xticks(), labels=8*ax.get_xticks())
    
if __name__ == "__main__":
    figs, ax = plt.subplots(3, 4)
    figs.set_size_inches(15, 10)

    stat = "edit_dist"
    
    for di in (glob.glob("data/eval/harmony_*_local_isl_isl_markov") +
               glob.glob("data/eval/harmony_*_SIP_None") + 
               glob.glob("data/eval/harmony_*_t5_None")):
        print(di)
        jj = int(re.search("harmony_(\\d)", di).group(1))
        if "markov" in di:
            ii = 0
        elif "SIP" in di:
            ii = 1
        elif "t5" in di:
            ii = 2

        print(di, ii, jj)

        titles = ["ae->aA", "a.e->a.A", "a.?e->a.?A", "a.*e->a.*A"]
        if ii == 0:
            title = titles[jj-1]
        else:
            title = ""

        rows = ["SIP-ISL", "SIP-FST", "t5"]
        if jj == 1:
            yt = rows[ii]
        else:
            yt = ""
            
        plotRecs(di + "/scores_new.tsv", ax[ii][jj - 1], title=title, yt=yt, stat=stat, xticks=(ii==2))
    #plt.show()
    figs.tight_layout()
    plt.savefig(f"plot_amp_{stat}.png")
