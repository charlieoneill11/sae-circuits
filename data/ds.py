import random
import einops
from mi_utils_public import *


model = from_pretrained("attn-only-4l")
model.cfg.use_attn_result = True

batch_size = 250
random.seed(0)
docstring_ind_prompt_kwargs = dict(
    n_matching_args=3,
    n_def_prefix_args=2,
    n_def_suffix_args=1,
    n_doc_prefix_args=0,
    met_desc_len=3,
    arg_desc_len=2,
)
prompts = [
    docstring_ind_prompt_gen("rest", **docstring_ind_prompt_kwargs)
    for _ in range(batch_size)
]
batched_prompts = BatchedPrompts(prompts=prompts, model=model)
clean_logits, clean_cache = model.run_with_cache(batched_prompts.clean_tokens)
corruptions = ["random_random"]
caches = {
    k: model.run_with_cache(batched_prompts.corrupt_tokens[k])[1] for k in corruptions
}
caches["clean"] = clean_cache
tokens = {k: batched_prompts.corrupt_tokens[k] for k in corruptions}
tokens["clean"] = batched_prompts.clean_tokens

clean_logits, clean_cache = model.run_with_cache(batched_prompts.clean_tokens)
head_resid, head_labels = clean_cache.stack_head_results(return_labels=True)
print(head_resid.shape)  # 32x250x41x512 = n_heads x batch_size x n_tokens x d_model
last_token_accum = head_resid.mean(dim=2).squeeze()
print(last_token_accum.shape)  # 32x250x512
last_token_accum = einops.rearrange(last_token_accum, "s b d -> b s d")
print(last_token_accum.shape)  # 250x32x512

corrupt_logits, corrupt_cache = model.run_with_cache(tokens["random_random"])
corrupt_head_resid, corrupt_head_labels = corrupt_cache.stack_head_results(
    return_labels=True
)
print(
    corrupt_head_resid.shape
)  # 32x250x41x512 = n_heads x batch_size x n_tokens x d_model
corrupt_last_token_accum = corrupt_head_resid.mean(dim=2).squeeze()
print(corrupt_last_token_accum.shape)  # 32x250x512
corrupt_last_token_accum = einops.rearrange(corrupt_last_token_accum, "s b d -> b s d")
print(corrupt_last_token_accum.shape)  # 250x32x512

# Concatenate them together and save
resid_streams = torch.cat([last_token_accum, corrupt_last_token_accum], dim=0)

ground_truth = [(0, 5), (1, 0), (1, 4), (2, 0), (3, 0), (3, 6)]

# Save
torch.save(resid_streams.to("cpu"), "../data/ds/resid_heads_mean.pt")
torch.save(ground_truth, "../data/ds/ground_truth.pt")
torch.save(head_labels, "../data/ds/labels_heads_mean.pt")
torch.save(prompts, "../data/ds/prompts_heads_mean.pt")
