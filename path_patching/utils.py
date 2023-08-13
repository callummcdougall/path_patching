import itertools
import einops
import torch as t

def product_with_args_kwargs(*args, **kwargs):
    '''
    Helper function which generates an iterable from args and kwargs.

    For example, running the following:
        product_with_args_kwargs([1, 2], ['a', 'b'], key1=[True, False], key2=["q", "k"])
    gives us a list of the following tuples:
        ((1, 'a'), {'key1', True, 'key2': 'q'})
        ((1, 'a'), {'key1', False})
        ((1, 'b'), ('key1', True))
        ...
        ((2, 'b'), {'key1', False, 'key2': 'k'})
    '''
    # Generate the product of args
    args_product = list(itertools.product(*args))
    
    # Generate the product of kwargs values
    kwargs_product = list(itertools.product(*kwargs.values()))
    
    # Combine args_product with each dict from kwargs_product
    result = []
    for args_values in args_product:
        for kwargs_values in kwargs_product:
            # Construct dict from keys and current values
            kwargs_dict = dict(zip(kwargs.keys(), kwargs_values))
            # Append the tuple with current args_values and kwargs_dict
            result.append((args_values, kwargs_dict))

    return result


def get_batch_and_seq_pos_indices(seq_pos, batch_size, seq_len):
    '''
    seq_pos can be given in four different forms:
        None             -> patch at all positions
        int              -> patch at this position for all sequences in batch
        list / 1D tensor -> this has shape (batch,) and the [i]-th elem is the position to patch for the i-th sequence in the batch
        2D tensor        -> this has shape (batch, pos) and the [i, :]-th elem are all the positions to patch for the i-th sequence
    
    This function returns batch_indices, seq_pos_indices as either slice(None)'s in the first case, or as 2D tensors in the other cases.

    In other words, if a tensor of activations had shape (batch_size, seq_len, ...), then you could index into it using:

        activations[batch_indices, seq_pos_indices]
    '''
    if seq_pos is None:
        seq_sub_pos_len = seq_len
        seq_pos_indices = slice(None)
        batch_indices = slice(None)
    else:
        if isinstance(seq_pos, int):
            seq_pos = [seq_pos for _ in range(batch_size)]
        if isinstance(seq_pos, list):
            seq_pos = t.tensor(seq_pos)
        if seq_pos.ndim == 1:
            seq_pos = seq_pos.unsqueeze(-1)
        assert (seq_pos.ndim == 2) and (seq_pos.shape[0] == batch_size) and (seq_pos.shape[1] <= seq_len) and (seq_pos.max() < seq_len), "Invalid 'seq_pos' argument."
        seq_sub_pos_len = seq_pos.shape[1]
        seq_pos_indices = seq_pos
        batch_indices = einops.repeat(t.arange(batch_size), "batch -> batch seq_sub_pos", seq_sub_pos=seq_sub_pos_len)
    
    return batch_indices, seq_pos_indices