import operator
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix

def is_valid(f, primary, supplemental):
    valid = True # assume true to start
    for x in supplemental:
        if x not in f.keys():
            valid = False
            break
    for x in primary:
        if isinstance(x, dict):
            if k not in f.keys():
                valid = False
                break
            for k,v in x.items():
                # occurs for ROI storage - v will be list of ROIs
                check = f[k]
                for val in v:
                    if val not in check.keys():
                        valid = False
                        break
        else:
            if x not in f.keys():
                valid = False
                break
    return valid

def create_comparison_function(operator_str):
    operators = {
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le,
        '==': operator.eq,
        '=': operator.eq
    }
    if operator_str in operators:
        return operators[operator_str]
    else:
        raise ValueError("Invalid comparison operator")

def _label_logic_from_config(config,filelist,supplementals):
    # TODO - test this haha, it's complicated
    logic = config.data.preprocessing.dynamic.endpoint.classify_logic
    neg_labels = pd.Series(index=filelist, data=False)
    pos_labels = pd.Series(index=filelist,data=False)
    neg = logic.negative
    pos = logic.positive
    for cat, holder in zip([neg, pos], [neg_labels,pos_labels]):
        for s in cat.scan():
            data = supplementals[s]
            subgroup = getattr(neg, s)
            fields = subgroup.scan()
            for field in fields:
                rule = getattr(subgroup, field)
                operator, value = rule.split(" ")
                value = float(value)
                comp_func = create_comparison_function(operator)
                rule_eval = data[field].apply(
                    lambda x: comp_func(x,value)
                    )
                holder = holder|rule_eval
    
    neg_labels = neg_labels * -1
    pos_labels = pos_labels * 1
    labels = neg_labels + pos_labels
    
    labels = labels.apply(lambda x: 99 if x == 0 else x)
    labels = labels.apply(lambda x: 0 if x == -1 else x)
    return labels
    
def rebuild_sparse(slices, rows, cols, refshape):
    dense = np.zeros(refshape)
    slice_nums = np.unique(slices).astype(int)
    for sl in slice_nums:
        slice_row_positions = rows[np.where(slices==sl)]
        slice_col_positions = cols[np.where(slices==sl)]
        sparse = coo_matrix(
            (np.ones_like(cols),(slice_row_positions,slice_col_positions)),
            shape=refshape[1:],
            dtype=int
            )
        dense[sl,...] = sparse.todense()
    return dense

def split_list(l, seqs):
    """Splits list l into sublists based on the fractions provided in seqs
    """
    results = []
    n_seqs = [round(s*len(l)) for s in seqs]
    for n in n_seqs:
        results.append(l[:n])
        l = l[n:]
    if len(l) != 0:
        results[-1] += l
    
    return results