"""Training Jez: the data mix, teacher labels and the fine-tune itself.

* :mod:`train.sources`       candidate datasets with licences and their mapping onto Jev questions
* :mod:`train.build_mix`     sample the mix, drop evaluation overlaps, write the manifest
* :mod:`train.teacher_label` soft labels from an open-weight teacher through the same label read-out
* :mod:`train.finetune`      full fine-tune (small) or LoRA (large) on label logits, with a learning-rate sweep
"""
