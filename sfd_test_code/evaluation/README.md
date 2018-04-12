## Getting WIDERFace mAP

For easy, medium and hard sets:

```
# Example:
python2.7 wider_eval.py -p ../output/WIDER_FACE/eval_tools/pred_orig -m ../output/WIDER_FACE/eval_tools/ground_truth/wider_easy_val.mat
```

Where `-p` is where the outputs of `test_wider.py` are and `-m` is where the groundtruth file is (can be `easy`, `medium` or `hard`). 

Run `wider_eval.py -h` for a help message.
