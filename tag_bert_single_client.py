"""Minimal demo: one honest client, BERT, TAG attack reconstructing its text.

Run with `python tag_bert_single_client.py` (add `dryrun` as an argument for a single-iteration smoke test).
"""

import sys
import logging

import torch

import breaching

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format="%(message)s")

dryrun = "dryrun" in sys.argv

cfg = breaching.get_config(overrides=["case=9_bert_training", "attack=tag"])
cfg.dryrun = dryrun

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))

cfg.case.user.num_data_points = 1  # one sentence
cfg.case.user.user_idx = 1  # the single client
cfg.case.data.shape = [10]  # sequence length
cfg.case.data.disable_mlm = True  # keep the target sentence unmasked
cfg.attack.optim.max_iterations = 100000  # raise (e.g. 12000) if recovery is poor

user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)

server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)
print("TRUE DATA:")
user.print(true_user_data)

# Gradient matching needs double-backward, which the fused SDPA kernels do not implement:
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.MATH):
    reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

# breaching.analysis.report needs datasets.load_metric, which current `datasets` releases no longer allow,
# so the two simple text metrics are computed directly here.
rec, true = reconstructed_user_data["data"].cpu(), true_user_data["data"].cpu()
position_accuracy = (rec == true).float().mean().item()
token_overlap = breaching.analysis.analysis.count_integer_overlap(
    rec.view(-1), true.view(-1), maxlength=cfg.case.data.vocab_size
).item()
print(f"Position-wise accuracy: {position_accuracy:.2%} | Token overlap (order-free): {token_overlap:.2%}")

print("RECONSTRUCTED DATA:")
user.print(reconstructed_user_data)
