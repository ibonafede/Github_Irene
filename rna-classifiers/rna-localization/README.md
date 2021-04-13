Optimization results:

- seqlist (split_by_window, window_size=4) + split 0.8/0.2
[2018-02-06 14:04:02.968207]  Training best config...
[2018-02-06 14:04:02.968242]  Configuration:  {'window_size': 14, 'epoch': 400, 'dimension': 150, 'neg': 2, 'word_ngrams': 4, 'learning_rate': 0.5}
Read 3M words
Number of words:  297
Number of labels: 3
Progress: 100.0% words/sec/thread:  330985 lr:  0.000000 loss:  0.125862 ETA:   0h 0m
[2018-02-06 14:09:36.488156]  Results:  N	311; P@1 0.701; R@1 0.701
[2018-02-06 14:09:37.409913]  Test set evaluation
[2018-02-06 14:09:38.187577]  Results:  N	389; P@1 0.769; R@1 0.769

- seqlist (split_by_pattern, seqlist & bearlist) + split 0.8/0.2
[2018-02-07 22:41:23.161623]  Best (configuration):  {'learning_rate': 1, 'dimension': 50, 'word_ngrams': 4, 'window_size': 8, 'epoch': 100, 'neg': 2}
[2018-02-07 22:41:23.161672]  Training best config...
[2018-02-07 22:41:23.161717]  Configuration:  {'learning_rate': 1, 'dimension': 50, 'word_ngrams': 4, 'window_size': 8, 'epoch': 100, 'neg': 2}
Read 0M words
Number of words:  67244
Number of labels: 2
Progress: 100.0% words/sec/thread:  541237 lr:  0.000000 loss:  0.116294 ETA:   0h 0m
[2018-02-07 22:41:36.999481]  Results:  N       301; P@1 0.797; R@1 0.797
[2018-02-07 22:41:37.292302]  Test set evaluation
[2018-02-07 22:41:37.450653]  Results:  N       377; P@1 0.785; R@1 0.785
