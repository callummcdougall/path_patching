from transformer_lens import HookedTransformer
from functools import partial
from path_patching.ioi_dataset import IOIDataset
import torch as t
from torch import Tensor
from jaxtyping import Float

# prompt_format = [
#     "When John and Mary went to the shops,{} gave the bag to",
#     "When Tom and James went to the park,{} gave the ball to",
#     "When Dan and Sid went to the shops,{} gave an apple to",
#     "After Martin and Amy went to the park,{} gave a drink to",
# ]
# name_pairs = [
#     (" John", " Mary"),
#     (" Tom", " James"),
#     (" Dan", " Sid"),
#     (" Martin", " Amy"),
# ]

# # Define 8 prompts, in 4 groups of 2 (with adjacent prompts having answers swapped)
# prompts = [
#     prompt.format(name)
#     for (prompt, names) in zip(prompt_format, name_pairs) for name in names[::-1]
# ]
# # Define the answers for each prompt, in the form (correct, incorrect)
# answers = [names[::i] for names in name_pairs for i in (1, -1)]
# # Define the answer tokens (same shape as the answers)
# answer_tokens = t.concat([
#     model.to_tokens(names, prepend_bos=False).T for names in answers
# ])

# def logits_to_ave_logit_diff(
#     logits: Float[Tensor, "batch seq d_vocab"],
#     answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
#     per_prompt: bool = False
# ):
#     '''
#     Returns logit difference between the correct and incorrect answer.

#     If per_prompt=True, return the array of differences rather than the average.
#     '''
#     # Only the final logits are relevant for the answer
#     final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
#     # Get the logits corresponding to the indirect object / subject tokens respectively
#     answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
#     # Find logit difference
#     correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
#     answer_logit_diff = correct_logits - incorrect_logits
#     return answer_logit_diff if per_prompt else answer_logit_diff.mean()

# clean_tokens = model.to_tokens(prompts, prepend_bos=True).to(device)
# flipped_indices = [i+1 if i % 2 == 0 else i-1 for i in range(len(clean_tokens))]
# flipped_tokens = clean_tokens[flipped_indices]

# clean_logits, clean_cache = model.run_with_cache(clean_tokens)
# flipped_logits, flipped_cache = model.run_with_cache(flipped_tokens)

# clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
# flipped_logit_diff = logits_to_ave_logit_diff(flipped_logits, answer_tokens)

# print(
#     "Clean string 0:    ", model.to_string(clean_tokens[0]), "\n"
#     "Flipped string 0:", model.to_string(flipped_tokens[0])
# )
# print(f"Clean logit diff: {clean_logit_diff:.4f}")
# print(f"Flipped logit diff: {flipped_logit_diff:.4f}")

# def ioi_metric_denoising(
#     logits: Float[Tensor, "batch seq d_vocab"],
#     answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
#     flipped_logit_diff: float = flipped_logit_diff,
#     clean_logit_diff: float = clean_logit_diff,
# ) -> Float[Tensor, ""]:
#     '''
#     Linear function of logit diff, calibrated so that it equals 0 when performance is
#     same as on flipped input, and 1 when performance is same as on clean input.
#     '''
#     patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
#     return ((patched_logit_diff - flipped_logit_diff) / (clean_logit_diff  - flipped_logit_diff)).item()

# labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0]))]




def _logits_to_ave_logit_diff(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()



def _ioi_metric_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float,
        corrupted_logit_diff: float,
        ioi_dataset: IOIDataset,
    ) -> float:
        '''
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        '''
        patched_logit_diff = _logits_to_ave_logit_diff(logits, ioi_dataset)
        return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)).item()



def generate_data_and_caches(N: int, model: HookedTransformer, verbose: bool = False, seed: int = 42, device: str = "cuda"):

    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device=str(device)
    )

    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYZ, BAB->XYZ")

    model.reset_hooks(including_permanent=True)

    ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
    abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

    ioi_average_logit_diff = _logits_to_ave_logit_diff(ioi_logits_original, ioi_dataset).item()
    abc_average_logit_diff = _logits_to_ave_logit_diff(abc_logits_original, ioi_dataset).item()

    if verbose:
        print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
        print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")

    ioi_metric_noising = partial(
        _ioi_metric_noising,
        clean_logit_diff=ioi_average_logit_diff,
        corrupted_logit_diff=abc_average_logit_diff,
        ioi_dataset=ioi_dataset,
    )

    return ioi_dataset, abc_dataset, ioi_cache, abc_cache, ioi_metric_noising