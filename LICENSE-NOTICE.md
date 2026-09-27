# Data License Notice

The FAQ data in `data/corpus.jsonl` is derived from the
**DFKI-NLP "faq-rewrites-llms" dataset**:
https://github.com/DFKI-NLP/faq-rewrites-llms

- Original data provided by **Deutsche Telekom AG** (German customer help page FAQs).
- Dataset compiled and released by **DFKI-NLP** (German Research Center for Artificial Intelligence).

## License

The dataset is released under the
**Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license:
https://creativecommons.org/licenses/by-sa/4.0/

`data/corpus.jsonl` is an adapted version of that dataset and is therefore also
distributed under CC BY-SA 4.0.

Changes made: only the human-written reference question/answer pairs were kept,
records were deduplicated by `instance_id` (the source repeats the same 56 FAQ
instances across 30 LLM-evaluation files), records with an identical
question+answer pair (ignoring whitespace differences) were merged into one
record listing all `source_instance_ids`, leading/trailing whitespace was
stripped, and the fields were renamed to `instance_id`, `question`, `answer`,
`use_case`. LLM-generated rewrites, inputs, and evaluation scores were dropped.

The CC BY-SA 4.0 license applies to the data only; the code in this repository
is licensed separately (see `LICENSE`).

## Citation

Citation as given in the source repository:

```bibtex
@inproceedings{gabryszak-etal-2024-enhancing-editorial,
    title = "Enhancing Editorial Tasks: A Case Study on Rewriting Customer Help Page Contents Using Large Language Models",
    author = {Gabryszak, Aleksandra  and
      R{\"o}der, Daniel  and
      Binder, Arne  and
      Sion, Luca  and
      Hennig, Leonhard},
    editor = "Mahamood, Saad  and
      Minh, Nguyen Le  and
      Ippolito, Daphne",
    booktitle = "Proceedings of the 17th International Natural Language Generation Conference",
    month = sep,
    year = "2024",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.inlg-main.33",
    pages = "402--411",
}
```
