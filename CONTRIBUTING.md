# Contributing

## Submitting benchmark results

This is the most useful thing you can contribute, and it takes about two minutes.

```bash
pip install quantcost
quantcost run
quantcost submit
```

`submit` validates the card, forks this repo, commits your result and opens the
pull request. If you do not have the [GitHub CLI](https://cli.github.com)
installed it prints a prefilled link instead, so you are never stuck.

### What makes a good submission

- **An idle machine.** Close the browser and the build you have running. If the
  report prints a *"varied by more than 15%"* warning, re-run before submitting.
- **Default settings.** `quantcost run` with no flags produces a card that
  is comparable with everyone else's. Non-default runs are still accepted and
  listed — they just are not ranked, because a 64-token run and a 32-token run
  do not measure the same thing.
- **Leave `--threads` alone** unless you are deliberately investigating it. The
  default pins ONNX Runtime to your machine's *performance* cores, and the card
  records how that was determined (`thread_policy`). This matters more than it
  sounds: on a heterogeneous CPU one thread scheduled onto an efficiency core
  gates the whole parallel region. Measured on an Apple M4 (4 performance + 6
  efficiency), pinning all ten physical cores cost int8 66% of its throughput
  and doubled run-to-run spread, while barely moving fp32. A card run across
  both tiers is measuring something different from everyone else's.
- **Interesting hardware especially welcome.** Raspberry Pi, old ThinkPads,
  Snapdragon laptops, bare-metal ARM servers, anything unusual. The point of the
  leaderboard is the *spread* across real machines, not the top of the list.

### What gets rejected, and why

CI runs `python -m edgellm.validate results/community` on every PR. It rejects
cards that are internally inconsistent (throughput that disagrees with the
latency and token count it claims), scored against a modified eval corpus, run
with an unpinned thread count, or carrying identifying information about your
machine. The intent is to keep the leaderboard comparable, not to accuse anyone:
if your card is rejected, the message says what to fix, and re-running almost
always fixes it.

Results are reviewed, not blindly trusted. Nothing here can prove a number came
from real silicon — but every input is pinned (model revision, artifact bytes,
corpus hash, prompt hash, token budget), so anyone can re-run the exact
configuration recorded in your card and compare.

### Privacy

A card contains your CPU model string, architecture, core count, RAM rounded to
the nearest gigabyte, OS name and release, and version strings. It does **not**
contain your hostname, username, file paths, IP or MAC address, or any machine
identifier. `edgellm/card.py::fingerprint` is the only function that reads your
machine, so you can check that claim in one place. The `--name` flag is optional
and the only thing that attaches an identity to a submission.

## Code changes

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

Adding the quantization/export path (PyTorch, Optimum, ONNX) needs the extra:

```bash
pip install -e ".[dev,quantize]"
```

Keep the default install torch-free. CI asserts that importing the CLI pulls in
no heavy dependency, because the contribution flow above depends on a fast
install. If you need torch, import it inside the function that uses it.
