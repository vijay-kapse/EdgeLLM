# Bundled evaluation corpus

`eval_wikitext2.txt` is a fixed 65,342-byte slice of the **WikiText-2 (raw)** test
split — the first 103 prose paragraphs, with blank lines and `= Section =` headings
removed, cut on a paragraph boundary.

- Source: <https://huggingface.co/datasets/Salesforce/wikitext>, config `wikitext-2-raw-v1`, split `test`
- License: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
- sha256: `34a2a43fcf49217f7a42133f961930d8c830ac7fe3cc7927f423f443083f97bf`

## Why it is bundled rather than downloaded

Perplexity is only comparable across machines if every machine scores
**byte-identical** text. Pinning the corpus into the package guarantees that, and
removes a heavy `datasets` dependency from the default install path. The hash is
checked at load time and recorded in every result card, so a submission computed
against a modified corpus is rejected rather than silently compared.
