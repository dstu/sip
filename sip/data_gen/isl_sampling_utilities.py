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

from sip.data_gen.gen_isl import select_factors, make_2isl_transducer, NotKISLError, replace_star_transitions, replace_star_state, print_fst, normalize_string, SYMBOL_RDELIM, SYMBOL_EPSILON
from sip.data_gen.utils import gen_pair, one_step, FSTCollection, random_subset, replace_arc, fst_to_json

def postprocess_for_sampling(fst: pynini.Fst):
    fst = replace_star_transitions(fst)
    if fst.state_names != None:
        fst = replace_star_state(fst)

    #check that the star replacement hasn't hosed the output charset
    osyms = fst.fst.output_symbols()
    for ii in range(osyms.num_symbols()):
        key = osyms.get_nth_key(ii)
        val = osyms.find(key)
        if val.startswith("("):
            val = ast.literal_eval(val)
            assert(len(val) <= 2)

    return fst

def apply_fst(fst, string):
    in_sym = fst.fst.input_symbols()
    out_sym = fst.fst.output_symbols()
    fst.fst.arcsort("ilabel")
    string = " ".join(string) + " </s>"
    acc = pynini.accep(string, token_type=in_sym)
    acc.set_input_symbols(in_sym)
    acc.set_output_symbols(in_sym)

    comp = pynini.compose(acc, fst.fst)
    result = normalize_string(comp.string(out_sym), delimiter="")
    return result

def fst_to_json(fst: pynini.Fst):
    s = []
    def in_to_str(label, syms):
        if label == 0:
            return chr(SYMBOL_EPSILON)
        else:
            ichar = syms.find(label)
            if ichar == "</s>":
                return chr(SYMBOL_RDELIM)
            else:
                assert(len(ichar) == 1)
                return ichar

    def out_to_str(label, syms):
        if label == 0:
            return chr(SYMBOL_EPSILON), chr(SYMBOL_EPSILON)
        else:
            ochar = syms.find(label)
            if ochar == "</s>":
                return chr(SYMBOL_RDELIM), chr(SYMBOL_EPSILON)
            elif ochar == "<e>":
                #should not happen b/c this should be label 0
                return chr(SYMBOL_EPSILON), chr(SYMBOL_EPSILON)                
            elif ochar.startswith("("):
                ois = ast.literal_eval(ochar)
                assert(len(ois) == 2)
                nois = []
                for ochar in ois:
                    if ochar == "<e>":
                        nois.append(chr(SYMBOL_EPSILON))
                    else:
                        assert(len(ochar) == 1)
                        nois.append(ochar)
                return nois
            else:
                assert(len(ochar) == 1)
                return ochar, chr(SYMBOL_EPSILON)

    def state_to_str(name):
        if name == "<e>":
            return chr(SYMBOL_EPSILON)
        else:
            if name == "</s>" or name == "<s>":
                 #use the same symbol for both edge markers
                return chr(SYMBOL_RDELIM)
            else:
                assert(len(name) == 1), f"Bad symbol !{name}!"
                return name
            
    state_names = fst.state_names
    fst = fst.fst
    assert fst.start() == 0

    for state in fst.states():
        for arc in fst.arcs(state):
            i1 = in_to_str(arc.ilabel, fst.input_symbols())
            o1, o2 = out_to_str(arc.olabel, fst.output_symbols())
            if state_names != None:
                sn_i = state_names[state]
                sn_o = state_names[arc.nextstate]
                assert(len(sn_i) == 1)
                assert(len(sn_o) == 1)
                sn_i = state_to_str(sn_i[0])
                sn_o = state_to_str(sn_o[0])
                s.append((sn_i, i1, o1, o2, sn_o))
            else:
                s.append((state, i1, o1, o2, arc.nextstate))
    return s

def gen_and_recode_pair(fst, isyms, osyms, **kwargs):
    #normal pair generator assumes internal char representation is utf8, but the ISL-fst uses its own symtab
    output_auto = pynini.randgen(fst, **kwargs)
    
    istr = []
    ostr = []
    for state in output_auto.topsort().states():
        arcs = list(output_auto.arcs(state))
        assert(len(arcs) <= 1)
        if arcs:
            arc = arcs[0]
            if arc.ilabel != 0: #length limit adds a ton of <e>:<e> transitions, which are the only input <e>s
                istr.append(isyms.find(arc.ilabel))
                if arc.olabel != 0: #trim output epsilons, leaving their inputs
                    ostr.append(osyms.find(arc.olabel))

    unicodeIstr = []
    for ii in istr:
        if ii == "</s>":
            unicodeIstr.append(chr(SYMBOL_RDELIM))
        else:
            unicodeIstr.append(ii)

    # print(istr)
    # print(ostr)

    unicodeOstr = []
    for oi in ostr:
        if oi == "</s>":
            unicodeOstr.append(chr(SYMBOL_RDELIM))
        elif oi.startswith("("):
            ois = ast.literal_eval(oi)
            ois = [ochar for ochar in ois if ochar != "<e>"]
            unicodeOstr += ois
        else:
            assert(oi != "<e>")
            unicodeOstr.append(oi)

    return "".join(unicodeIstr), "".join(unicodeOstr)

def recode_string(unicode_str, vocab):
    fst = pynini.Fst()
    fst.set_input_symbols(vocab)
    fst.set_output_symbols(vocab)
    fst.add_states(len(unicode_str) + 1)

    for ind, ch in enumerate(unicode_str):
        if ord(ch) == SYMBOL_RDELIM:
            ch = "</s>"
        fst.add_arc(ind,
                    pynini.Arc(vocab.find(ch),
                               vocab.find(ch),
                               0,
                               ind + 1))
    fst.set_final(len(unicode_str))
    fst.set_start(0) #thought this was required? but apparently not
    return fst

class DataGen:
    def __init__(self, fst, chosen_vocab, vocab):
        self.fst = fst
        self.sampling_fst = postprocess_for_sampling(fst)
        self.train_length = one_step(self.sampling_fst.fst.input_symbols()).closure(3, 15)
    def gen_train_ex(self) -> tuple[str, str]:
        raise NotImplementedError()

    def gen_test_ex(self) -> tuple[str, str]:
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

class IidGen(DataGen):

    def __init__(self, fst, chosen_vocab, vocab):
        super().__init__(fst, chosen_vocab, vocab)
        self.train_length_restricted = pynini.compose(self.train_length, self.sampling_fst.fst)

    def gen_train_ex(self):
        return gen_and_recode_pair(self.train_length_restricted,
                                   self.sampling_fst.fst.input_symbols(), self.sampling_fst.fst.output_symbols(),
                                   seed=random.randint(0, 2**31-1))

    def gen_test_ex(self) -> tuple[str, str]:
        return self.gen_train_ex()

    @property
    def name(self):
        return "iid"


class LengthGen(DataGen):
    def __init__(self, fst, chosen_vocab, vocab):
        super().__init__(fst, chosen_vocab, vocab)
        k = 3 # deepest recursion allowed in training
        self.train_data, self.test_data = self.create_recursion_split(self.sampling_fst,
                                                                      vocab, range(2, 31), range(k+1), range(k+1, 31))
        random.shuffle(self.train_data)
        random.shuffle(self.test_data)
        self.train_index = 0
        self.test_index = 0

    @staticmethod
    def create_recursion_split(fst: pynini.Fst, vocab, length_range, train_range, test_range, samples=5_000):
        assert len(set(train_range) & set(test_range)) == 0

        train_data = []
        test_data = []
        fst_faster_access = FasterFSTAccess(fst)
        for l in length_range:
            length_l = pynini.compose(one_step(vocab).closure(l, l), fst)
            for _ in range(samples):
                i, o = gen_pair(length_l, seed=random.randint(0, 2 ** 31 - 1))
                max_recursion_count = max(count_state_visits(fst_faster_access, i))
                if max_recursion_count in train_range:
                    train_data.append((i, o))
                elif max_recursion_count in test_range:
                    test_data.append((i, o))
        return train_data, test_data

    def gen_train_ex(self) -> tuple[str, str]:
        i, o = self.train_data[self.train_index]
        self.train_index += 1
        return i, o

    def gen_test_ex(self) -> tuple[str, str]:
        i, o = self.test_data[self.test_index]
        self.test_index += 1
        return i, o

    @property
    def name(self):
        return "length"


class UnseenCombinationsGen(DataGen):
    def __init__(self, fst, chosen_vocab, vocab):
        super().__init__(fst, chosen_vocab, vocab)
        self.train_length = one_step(chosen_vocab).closure(5, 20)
        train_fst, test_fst, (self.l, self.r) = gen_compgen_variants(fst, 20)
        self.train_fst = pynini.compose(self.train_length, postprocess_for_sampling(train_fst, chosen_vocab, vocab))
        self.test_fst = pynini.compose(self.train_length, postprocess_for_sampling(test_fst, chosen_vocab, vocab))

    def gen_train_ex(self) -> tuple[str, str]:
        i, o = gen_pair(self.train_fst, seed=random.randint(0, 2**31-1))
        restr_i = pynini.compose(pynini.accep(i, token_type="utf8"), self.sampling_fst)
        test_fst_restr = pynini.compose(pynini.accep(i, token_type="utf8"), self.test_fst)
        assert restr_i.num_states() > 0
        assert test_fst_restr.num_states() == 0, "Training example should not be accepted by test FST"
        return i, o

    def gen_test_ex(self) -> tuple[str, str]:
        i, o = gen_pair(self.test_fst, seed=random.randint(0, 2**31-1))
        restr_i = pynini.compose(pynini.accep(i, token_type="utf8"), self.sampling_fst)
        train_fst_restr = pynini.compose(pynini.accep(i, token_type="utf8"), self.train_fst)
        assert restr_i.num_states() > 0
        assert train_fst_restr.num_states() == 0, "Test example should not be accepted by training FST"
        return i, o

    @property
    def name(self):
        return "uc"
