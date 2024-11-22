import random
from collections import *
import copy
import itertools

import pynini, pywrapfst

class NotKISLError(Exception):
    def __init__(self):
        pass

class ISLTransducer:
    def __init__(self, fst):
        self.fst = fst
        self.uninteresting_chars = None
        self.state_names = None

    def copy(self):
        xcopy = ISLTransducer(self.fst.copy())
        xcopy.state_names = copy.copy(self.state_names)
        xcopy.uninteresting_chars = copy.copy(self.uninteresting_chars)
        return xcopy

def choose_input_char(alphabet, xi, f_len):
    if xi == 0:
        return random.choice(list(alphabet) + ["<s>"])
    elif xi == f_len - 1:
        return random.choice(list(alphabet) + ["</s>"])
    else:
        #string terminator is not allowed in the interior of the string
        return random.choice(alphabet)

def transduce_terminator(xi):
    if xi in ["<s>", "</s>"]:
        return "<e>"
    else:
        return xi

def select_factors(n_factors, factor_length, alphabet, p_progressive:float=0, p_regressive:float=0,
                   epsilon_allowed:bool=True, p_epenthesis:float=0):
    factors_in = set()
    factors = []

    if epsilon_allowed:
        output_alphabet = list(alphabet) + ["<e>"]
    else:
        output_alphabet = list(alphabet)
    
    while len(factors) < n_factors:
        f_len = random.randint(2, factor_length)

        epenthesis_type = None
        if random.random() < p_epenthesis:
            #although a k-isl language doesn't have to have output sequences of at most length k
            #this is convenient for a fixed-dimension vector encoding
            #to preserve this, epenthetic patterns must either be progressive (a b) -> (a cd)
            #or have input-side factors of length k-1 (a) -> (b a)
            if random.random() < p_progressive:
                epenthesis_type = "progressive"
                p_progressive = 1
            else:
                epenthesis_type = "short"
                f_len -= 1

        fx = tuple([choose_input_char(alphabet, xi, f_len) for xi in range(f_len)])
        #factors must be unique on input side so transduction is deterministic
        if fx not in factors:
            factors_in.add(fx)

            if random.random() < p_progressive:
                if epenthesis_type == "progressive":
                    f_out = fx[:-1] + ( (random.choice(output_alphabet), random.choice(output_alphabet)), )
                else:
                    f_out = fx[:-1] + (random.choice(output_alphabet),) 
            elif random.random() < p_regressive:
                f_out = (random.choice(output_alphabet),) + fx[1:]
            else:
                f_out = tuple([random.choice(output_alphabet) for xi in range(f_len)])

            if epenthesis_type == "short":
                index = random.randrange(len(f_out))
                if random.random() < .5:
                    n_r = (f_out[index], random.choice(output_alphabet))
                else:
                    n_r = (random.choice(output_alphabet), f_out[index])
                f_new = list(f_out)
                f_new[index] = n_r
                f_out = tuple(f_new)

            f_out = tuple([transduce_terminator(xi) for xi in f_out])
            assert(len(fx) == len(f_out))
            factors.append((fx, f_out))

    return factors

def char_seqs(alphabet, factor_length:int):
    if factor_length == 0:
        yield tuple()
        return

    for char in alphabet:
        for ch_seq in char_seqs(alphabet, factor_length - 1):
            yield (char,) + ch_seq

def has_outgoing(arcs, char):
    for ai in arcs:
        if ai.ilabel == char:
            return True
    return False

def flatten(seq):
    n_s = []
    for item in seq:
        if type(item) != str and hasattr(item, "__iter__"):
            n_s += item
        else:
            n_s.append(item)
    return n_s

def replace(seq, factors):
    new_seq = [xi for xi in seq]
    for (in_s, out_s) in factors:
        for pos, _ in enumerate(seq):
            sub_s = seq[pos:pos + len(in_s)]
            if sub_s == in_s:
                for pos2, repl in enumerate(out_s):
                    if repl != in_s[pos2]:
                        new_seq[pos + pos2] = repl

    new_seq = [xx for xx in new_seq if xx != "<e>"]
    return flatten(new_seq)

def lcp(strings):
    m_len = min([len(xi) for xi in strings])
    for pref_len in range(m_len, -1, -1):
        prefs = [tuple(xi[:pref_len]) for xi in strings]
        if len(set(prefs)) == 1:
            return prefs[0]

class Trie:
    def __init__(self, node=("<s>",)):
        self.node = node
        self.output = tuple()
        self.children = {}

    def add(self, seq, output):
        if len(seq) == 0:
            self.output = output
        else:
            if seq[0] == "<s>":
                seq = seq[1:]

            if seq[0] not in self.children:
                self.children[seq[0]] = Trie(self.node + (seq[0],))
            self.children[seq[0]].add(seq[1:], output)

    def move_lcp(self):
        for chi in self.children.values():
            chi.move_lcp()

        if len(self.children) == 0:
            return

        outputs = [ch.output for ch in self.children.values()]
        lcpS = lcp(outputs)
        if len(lcpS) > 0 and lcpS[-1] == "</s>":
            lcpS = lcpS[:-1]
        # print("lcp of", outputs, "is", lcpS)
        self.output = lcpS
        for ch in self.children.values():
            ch.output = ch.output[len(lcpS):]

    def transitions(self):
        transitions = {}

        for char, chi in self.children.items():
            output = chi.output
            if self.node == ("<s>",):
                #start state doesn't have an incoming transition
                output = self.output + tuple(output)
            output = tuple([xx for xx in output if xx not in ("<s>", "</s>")])
            transitions[char] = (output, chi.node)
        return transitions
            
    def states(self):
        if not self.children:
            return

        yield self
        for chi in self.children.values():
            for si in chi.states():
                yield si
            
    def print(self, depth=0):
        print("\t"*depth, self.node, ":", self.output)
        for chi in self.children.values():
            chi.print(depth+1)

def prefix_all(trans_tab, state, prefix):
    # print("\t", "prefixing outgoing transitions of", state, "with", prefix)

    tt = trans_tab[state]
    for char, (out, dest) in tt.items():
        new_out = prefix + out
        tt[char] = (new_out, dest)

def merge_transitions(merge_from, merge_to, state_tab, trans_tab, indices=False):
    if not indices:
        # print("merging", state_tab[merge_from], state_tab[merge_to])
        from_ind = state_tab[merge_from]
        to_ind = state_tab[merge_to]
    else:
        from_ind = merge_from
        to_ind = merge_to

    if from_ind == to_ind:
        return

    for state, ind in state_tab.items():
        if ind == from_ind:
            state_tab[state] = to_ind

    trans_to = trans_tab[to_ind]

    if from_ind in trans_tab:
        trans_from = trans_tab[from_ind]
        del trans_tab[from_ind]
    else:
        trans_from = {}

    # print("\t", trans_from)
    # print("\t", trans_to)
        
    for ind, trans in trans_tab.items():
        n_trans = {}
        for char, (out, old_index) in trans.items():
            dest = old_index
            if old_index == from_ind:
                dest = to_ind
            n_trans[char] = (out, dest)
        trans_tab[ind] = n_trans

    trans_to = trans_tab[to_ind]

    for char, (out_from, dest_from) in trans_from.items():
        if char in trans_to:
            out_to, dest_to = trans_to[char]
            if out_from != out_to or dest_from != dest_to:
                # print("non-determinism: merged state now has", char, out_to, dest_to, out_from, dest_from)
                #pushback: retain lcp, prefix what's left to all outgoing transitions from dest. stts
                lc = lcp([out_to, out_from])
                res_to = out_to[len(lc):]
                res_from = out_from[len(lc):]
                if char == "</s>" and max(len(res_to), len(res_from)) > 0:
                    raise NotKISLError()

                prefix_all(trans_tab, dest_from, res_from)
                prefix_all(trans_tab, dest_to, res_to)
                trans_to[char] = (lc, dest_to)
                # print("\tinitiating recursive merge of", dest_from, dest_to)
                merge_transitions(dest_from, dest_to, state_tab, trans_tab, indices=True)
        else:
            trans_to[char] = (out_from, dest_from)

def merge_states(trie):
    state_tab = { ("</s>",) : 0 }
    trans_tab = { 0 : {} }
    for state in trie.states():
        trans = state.transitions()
        # print("N:", state.node)
        # print(trans)
        # print()

        state_tab[state.node] = len(state_tab)        
        trans_tab[state_tab[state.node]] = trans

        destination = state.node[-1:]
        if destination in state_tab or len(destination) == 0:
            continue
        state_tab[destination] = len(state_tab)
        trans_tab[state_tab[destination]] = {}

    for ind, trans in trans_tab.items():
        n_trans = {}
        for char, (out, dest) in trans.items():
            if dest[-1] == "</s>":
                dest_index = state_tab[("</s>",)]
            else:
                if dest not in state_tab:
                    state_tab[dest] = len(state_tab)
                dest_index = state_tab[dest]
            # print("translating", dest, "into", dest_index)
            n_trans[char] = (out, dest_index)
        trans_tab[ind] = n_trans

    # print("dstate")
    # for si, vi in state_tab.items():
    #     print(si, vi)

    rstate = {}
    for state, ind in state_tab.items():
        if ind not in rstate or len(state) < len(rstate[ind]):
            rstate[ind] = state

    # print("dtrans")
    # for ind, trans in trans_tab.items():
    #     print(ind, rstate[ind])
    #     for kk, vv in trans.items():
    #         print("\t", kk, vv, rstate[vv[1]])

    for state, ind in sorted(state_tab.items(), key=lambda xx: (len(xx[0]), xx[0])):
        key = state[-1:]
        if key == state:
            continue

        # print("merge", state, "with", key)
        merge_transitions(state, key, state_tab, trans_tab)

    rstate = {}
    for state, ind in state_tab.items():
        if ind not in rstate or len(state) < len(rstate[ind]):
            rstate[ind] = state

    # for ind, trans in trans_tab.items():
    #     print(ind, rstate[ind])
    #     for kk, vv in trans.items():
    #         print("\t", kk, vv[0], vv[1], rstate[vv[1]])

    return state_tab, trans_tab

def replace_stars(char, out, alphabet):
    def replace(cx, rr):
        # if cx == "*":
        #     return rr
        # return cx
        return cx.replace("*", rr)

    if char == "*" or "*" in out:
        for repl in alphabet:
            n_char = replace(char, repl)
            if isinstance(out, str):
                n_out = replace(out, repl)
            else:
                n_out = tuple([replace(ci, repl) for ci in out])
            yield n_char, n_out
    else:
        yield char, out

def replace_star_transitions(fst):
    fst = fst.copy()
    isyms = pynini.SymbolTable()
    old_isyms = fst.fst.input_symbols()
    for ii in range(old_isyms.num_symbols()):
        key = old_isyms.get_nth_key(ii)
        val = old_isyms.find(key)
        isyms.add_symbol(val)

    osyms = pynini.SymbolTable()
    old_osyms = fst.fst.output_symbols()
    for ii in range(old_osyms.num_symbols()):
        key = old_osyms.get_nth_key(ii)
        val = old_osyms.find(key)
        osyms.add_symbol(val)

    copied_arcs = {}
    for state in fst.fst.states():
        copied_arcs[state] = list(fst.fst.arcs(state))

    for state, arcs in copied_arcs.items():
        fst.fst.delete_arcs(state)

        for arc in arcs:
            char = old_isyms.find(arc.ilabel)
            rewrite = old_osyms.find(arc.olabel)
            for char_c, rewrite_c in replace_stars(char, rewrite, fst.uninteresting_chars):
                if rewrite_c == tuple():
                    rewrite_c = "<e>"
                if len(rewrite_c) == 1:
                    rewrite_c = rewrite_c[0]

                fst.fst.add_arc(state,
                                pynini.Arc(isyms.add_symbol(char_c),
                                           osyms.add_symbol(str(rewrite_c)),
                                           0,
                                           arc.nextstate))

    fst.fst.set_input_symbols(isyms)
    fst.fst.set_output_symbols(osyms)

    return fst

def replace_star_state(fst):
    fst = fst.copy()
    nstate = {}
    mapping = {}
    old_states = list(fst.fst.states())
    for ind, state in fst.state_names.items():
        nstate[state] = ind
        mapping[ind] = [ind,]

    def new_state():
        while True:
            fst.fst.add_state()
            yield len(nstate)

    for ind, state in fst.state_names.items():
        if "*" in state:
            del nstate[state]
            mapping[ind] = []
            names = itertools.chain([ind,], new_state())
            for _, new_state in replace_stars("", state, fst.uninteresting_chars):
                new_name = next(names)
                nstate[new_state] = new_name
                mapping[ind].append(nstate[new_state])

    copied_arcs = {}
    for state in old_states:
        copied_arcs[state] = list(fst.fst.arcs(state))

    isyms = fst.fst.input_symbols()

    for state, arcs in copied_arcs.items():
        fst.fst.delete_arcs(state)

        for m_src in mapping[state]:
            for arc in arcs:
                char = isyms.find(arc.ilabel)
                m_dest = nstate[(char,)]
                fst.fst.add_arc(m_src,
                                pynini.Arc(arc.ilabel,
                                           arc.olabel,
                                           0,
                                           m_dest))

    fst.state_names = nstate

    return fst

def renumber_states(states, trans):
    nstate = { ("<s>",) : 0,
               ("</s>",) : 1, }
    mapping = {}

    rstate = {}
    for state, ind in states.items():
        if ind not in rstate or len(state) < len(rstate[ind]):
            rstate[ind] = state

    for ind in trans:
        name = rstate[ind]
        mapping[ind] = []

        if name not in nstate:
            nstate[name] = len(nstate)

        mapping[ind] = nstate[name]

    # print("renumbering", mapping)
            
    ntrans = {}
    for ind, tt in trans.items():
        m_src = mapping[ind]
        ntrans[m_src] = {}
        for char, (out, dest) in tt.items():
            if out == tuple():
                out = "<e>"
            if len(out) == 1:
                out = out[0]

            ntrans[m_src][char] = (out, mapping[dest])

    rstate = {}
    for state, ind in nstate.items():
        rstate[ind] = state

    return nstate, rstate, ntrans

def make_2isl_transducer(factors, alphabet, minimize:bool=True):
    # print("Making ISL transducer for factor set:", factors)

    important_chars = set(["*"])
    for (in_s, out_s) in factors:
        important_chars.update(in_s)
        important_chars.update([ci for ci in out_s if type(ci) == str])

    important_chars.discard("<e>")
    important_chars.discard("<s>")
    important_chars.discard("</s>")

    trie = Trie()

    for ln in range(4):
        for seq in char_seqs(important_chars, ln):
            seq = ("<s>",) + seq + ("</s>",)
            # print(seq, replace(seq, factors))
            trie.add(seq, replace(seq, factors))
        
    # trie.print()

    trie.move_lcp()

    # print("after lcp move")
    # trie.print()
    # print("--")

    state, trans = merge_states(trie)
    # print(trans)
    # print("renum")
    state, rstate, trans = renumber_states(state, trans)
    # print(state)
    # print(rstate)
    # print(trans)

    fst = ISLTransducer(pynini.Fst())
    input_alphabet = pynini.SymbolTable()
    output_alphabet = pynini.SymbolTable()
    input_alphabet.add_symbol("<e>", key=0)
    output_alphabet.add_symbol("<e>", key=0)

    for ind, tt in trans.items():
        for char, (out, dest) in tt.items():
            #print("adding", char, out)
            input_alphabet.add_symbol(char)
            output_alphabet.add_symbol(str(out))

    fst.fst.set_input_symbols(input_alphabet)
    fst.fst.set_output_symbols(output_alphabet)

    fst.fst.add_states(len(trans))
    for ind, trans in trans.items():
        for char, (out, dest) in trans.items():
            fst.fst.add_arc(ind,
                            pynini.Arc(input_alphabet.find(char),
                                       output_alphabet.find(str(out)),
                                       0,
                                       dest))
    fst.fst.set_start(0)
    fst.fst.set_final(1)

    fst.uninteresting_chars = set(alphabet).difference(important_chars)
    fst.state_names = rstate

    if minimize:
        copied_fst = fst.copy()
        fst.fst = fst.fst.minimize(allow_nondet=False)
        # Check that minimization didn't introduce any transitions that have epsilon on the input tape
        minimized_ok = True
        for state in fst.fst.states():
            for arc in fst.fst.arcs(state):
                if input_alphabet.find(arc.ilabel) == "<e>":
                    minimized_ok = False
                    break
        if not minimized_ok:
            # print("Failed to minimize")
            fst.fst = copied_fst
        else:
            fst.state_names = None

    return fst

def print_fst(fst):
    in_sym = fst.fst.input_symbols()
    out_sym = fst.fst.output_symbols()

    for ind in fst.fst.states():
        print(ind)
        for arc in fst.fst.arcs(ind):
            print(f"\t{arc.ilabel}:{arc.olabel} -> {arc.nextstate}",
                  f"\t{in_sym.find(arc.ilabel)}:{out_sym.find(arc.olabel)}")

def test_fst(fst, string, expected=None):
    fst = replace_star_transitions(fst)
    # print("After replacement")
    # print("uninteresting set", fst.uninteresting_chars)
    # print_fst(fst)
    if fst.state_names != None:
        fst = replace_star_state(fst)
        # print("After state repl")
        # print_fst(fst)

    in_sym = fst.fst.input_symbols()
    out_sym = fst.fst.output_symbols()
    fst.fst.arcsort("ilabel")
    string = " ".join(string) + " </s>"
    acc = pynini.accep(string, token_type=in_sym)
    acc.set_input_symbols(in_sym)
    acc.set_output_symbols(in_sym)
    
    comp = pynini.compose(acc, fst.fst)
    result = normalize_string(comp.string(out_sym))
    print(result)
    if expected != None and result != " ".join(expected):
        print("TEST FAILED: expected", " ".join(expected))
        print("Unnormalized:", comp.string(out_sym))

def normalize_string(string):
    nStr = string.replace("(", "").replace(")", "").replace(",", "").replace("'", "").replace("<e>", "")
    return " ".join(nStr.split())
    
if __name__ == "__main__":
    alphabet = "abcdefg"
    factors = select_factors(5, 2, alphabet, p_progressive=0, p_regressive=0)
    print("Random 2-Factors", factors)
    factors = select_factors(5, 2, alphabet, p_progressive=1, p_regressive=0)
    print("Random progressive 2-Factors", factors)
    factors = select_factors(5, 2, alphabet, p_progressive=0, p_regressive=1)
    print("Random regressive 2-Factors", factors)
    
    factors = select_factors(5, 3, alphabet, p_progressive=0, p_regressive=0)
    print("Random 3-Factors", factors)
    factors = select_factors(5, 3, alphabet, p_progressive=1, p_regressive=0)
    print("Random progressive 3-Factors", factors)
    factors = select_factors(5, 3, alphabet, p_progressive=0, p_regressive=1)
    print("Random regressive 3-Factors", factors)

    factors = select_factors(5, 2, alphabet, p_progressive=.5, p_regressive=0, p_epenthesis=1)
    print("Random epenthetic 2-Factors", factors)
    
    print("Identity transducer:")
    fst = make_2isl_transducer([], "abc")
    print_fst(fst)
    test_fst(fst, "ababccc", "ababccc")
    print()

    print("Simple progressive pattern:")
    print()
    fst = make_2isl_transducer(
        [
            (("a", "b"), ("a", "c")),
        ],
        "abc")
    print_fst(fst)
    test_fst(fst, "ababc", "acacc")
    test_fst(fst, "ababcc", "acaccc")
    print()

    print("Simple regressive pattern:")
    fst = make_2isl_transducer(
        [
            (("a", "b"), ("c", "b")),
        ],
        "abc")
    print_fst(fst)
    test_fst(fst, "b", "b")
    test_fst(fst, "ab", "cb")
    test_fst(fst, "ac", "ac")
    test_fst(fst, "abaacbab", "cbaacbcb")
    print()

    print("Two-char substitution:")
    fst = make_2isl_transducer(
        [
            (("a", "b"), ("c", "c")),
        ],
        "abc")
    print_fst(fst)
    test_fst(fst, "b", "b")
    test_fst(fst, "ab", "cc")
    test_fst(fst, "ac", "ac")
    test_fst(fst, "abaacbab", "ccaacbcc")
    print()

    print("Character deletion:")
    fst = make_2isl_transducer(
        [
            (("a", "b"), ("a", "<e>")),
        ],
        "abc")
    print_fst(fst)
    test_fst(fst, "a", "a")
    test_fst(fst, "b", "b")
    test_fst(fst, "aaba", "aaa")
    test_fst(fst, "ababc", "aac")
    print()

    print("Character epenthesis:")
    fst = make_2isl_transducer(
        [
            (("a", "b"), ("a", ("a", "c"),)),
        ],
        "abc")
    print_fst(fst)
    test_fst(fst, "ab", "aac")
    test_fst(fst, "b", "b")
    test_fst(fst, "aaba", "aaaca")
    print()

    print("Epenthesis before trigger:")
    fst = make_2isl_transducer(
        [
            (("a",), (("c", "a"),)),
        ],
        "abc")
    print_fst(fst)
    test_fst(fst, "a", "ca")
    test_fst(fst, "b", "b")
    test_fst(fst, "aaba", "cacabca")
    print()

    print("Prefix to string:")
    fst = make_2isl_transducer(
        [
            (("<s>",), ("c",)),
        ],
        "abc")
    print_fst(fst)
    test_fst(fst, "a", "ca")
    test_fst(fst, "b", "cb")
    test_fst(fst, "aaba", "caaba")
    print()

    print("Suffix to string:")
    fst = make_2isl_transducer(
        [
            (("</s>",), ("c",)),
        ],
        "abc")
    print_fst(fst)
    test_fst(fst, "a", "ac")
    test_fst(fst, "b", "bc")
    test_fst(fst, "aaba", "aabac")
    print()

    print("Illegitimate pattern (same character twice in input):")
    try:
        fst = make_2isl_transducer(
            [
                (("a", "a"), ("c", "a")),
                (("a", "a"), ("a", "c")),
            ],
            "abc")
    except NotKISLError:
        print("Language is not 2ISL")
    print()

    print("Illegitimate pattern with deletion:")
    try:
        fst = make_2isl_transducer(
            [
                (("a", "a"), ("<e>", "<e>")),
            ],
            "abc")
    except NotKISLError:
        print("Language is not 2ISL")

    #check that string suffixation is handled correctly
