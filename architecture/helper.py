from fastai.text import *
import pandas as pd
from functools import wraps

def clean_impressions(data,column,to_remove):
    data[column] = data[column].str.strip()
    for item in to_remove:
        data[column] = data[column].str.replace(item, '')
    return data


def shuffle_split_dataset(data,train_frac=.85,val_frac=.1,test_frac=.05,seed=42):
    
    data = data.sample(frac=1,random_state=seed)
    train_index = int(data.count()[0]*train_frac)
    val_index = train_index + int(data.count()[0]*val_frac)
    
    train_set = data[:train_index]
    val_set = data[train_index:val_index]
    test_set = data[val_index:]
    return train_set, val_set, test_set


def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        return func 
    return decorator


def _get_processor(tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                   min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
    return [TokenizeProcessor(tokenizer=tokenizer, chunksize=chunksize, 
                              mark_fields=mark_fields, include_bos=include_bos, include_eos=include_eos),
            NumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]


@add_method(TextLMDataBunch)
def report_slicer(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, chunksize:int=10000, max_vocab:int=60000,
                min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, **kwargs) -> DataBunch:
    
        processor = _get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, mark_fields=mark_fields, 
                                   include_bos=include_bos, include_eos=include_eos)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        if cls==TextLMDataBunch: src = src.label_for_lm()
        else: 
            if label_delim is not None: src = src.label_from_df(cols=label_cols, classes=classes, label_delim=label_delim)
            else: src = src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)
    
    
@add_method(TextClasDataBunch)
def create_labeled_set(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
            tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
            label_cols:IntsOrStrs=0, label_delim:str=None, chunksize:int=10000, max_vocab:int=60000,
            min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, **kwargs) -> DataBunch:
    processor = _get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                               min_freq=min_freq, mark_fields=mark_fields, 
                               include_bos=include_bos, include_eos=include_eos)
    if classes is None and is_listy(label_cols) and len(label_cols) > 1: 
        classes = label_cols
    src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                    TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
    if cls==TextLMDataBunch: 
        src = src.label_for_lm()
    else: 
        if label_delim is not None: 
            src = src.label_from_df(cols=label_cols, classes=classes, label_delim=label_delim)
        else: src = src.label_from_df(cols=label_cols, classes=classes)
    if test_df is not None: 
        src.add_test(TextList.from_df(test_df, path, cols=text_cols))
    return src.databunch(**kwargs)