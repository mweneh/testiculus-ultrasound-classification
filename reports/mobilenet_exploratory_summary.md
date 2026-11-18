# Exploratory MobileNetV3-Small comparison

MobileNet was added after the original Custom CNN and ResNet18 test results
were known. This is a post-hoc exploratory extension, not an independent
confirmatory comparison. Original models, thresholds and results are unchanged.

## CPU training and validation

Device: `cpu`. ImageNet initialization, frozen features and
frozen feature BatchNorm statistics; classification head optimized with AdamW
and training-label class-weighted cross-entropy. Learning rate 0.0001, batch
size 16, maximum 20 epochs, patience 5, seed 42, two CPU threads, no loader workers.
Completed 10 epochs in 93.44 seconds;
best epoch 5, validation loss 0.541681.
Epoch durations: median 9.25 seconds,
range 8.83–10.19 seconds.
Training duration includes training/validation epochs and checkpoint/history I/O;
it excludes initialization/download and dataset construction. Full epoch metrics
and durations are preserved in `reports/mobilenet_exploratory_results.json`.
Trainable parameters: 592,898 / 1,519,906.

No additional fine-tuning: retain the declared frozen-backbone experiment. Validation evidence does not establish clear underfitting requiring an additional run.

Validation uses 129 images (104 homogeneous, 25 inhomogeneous) and deterministic
224×224 bilinear resizing, grayscale repeated to three channels and ImageNet
normalization. The best checkpoint is selected by minimum validation loss.
Threshold candidates are every unique validation probability; maximize
positive-class F1 and choose the highest threshold on exact ties.

Frozen threshold: `0.5038942694664001`.

| Validation ROC-AUC | AP | Sensitivity | Specificity | F1 | Balanced accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.6846 | 0.4631 | 0.4000 | 0.9423 | 0.4878 | 0.6712 |

The checkpoint, validation predictions and selected threshold were hashed and
frozen before one guarded test evaluation. Reusing its output directory is
refused, including after a partial failure. No test-based tuning followed.

## Measured efficiency and test results

Hardware: Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz; Linux-6.8.0-149-generic-x86_64-with-glibc2.39.
Python 3.12.3, PyTorch 2.14.1+cpu,
Torchvision 0.29.1+cpu; CPU float32, batch size 1,
224×224 inputs, 2 intra-op / 1 inter-op threads.
Each model uses 10 warm-up forwards and 50 timed forwards.
The same synthetic grayscale tensor is used with each model's channel and
normalization conventions. Latency measures model forward under inference_mode;
image decoding, preprocessing, disk I/O and API overhead are excluded.
State size is the sum of state_dict tensor bytes, including buffers; checkpoint
size is the actual saved file. MiB = 1,048,576 bytes. Timings are hardware-specific.

All test metrics use the same saved sample identities (129 images, 104/25).
Rows for the original models reuse verified Colab metrics without rerunning them.

| Model | Parameters | State MiB | Checkpoint MiB | Median CPU ms | Test ROC-AUC | Test AP | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN (original) | 389,314 | 1.489 | 1.499 | 51.465 | 0.6123 | 0.4029 | 0.4000 | 0.6635 | 0.2857 |
| ResNet18 (original) | 11,177,538 | 42.676 | 42.713 | 62.170 | 0.7473 | 0.5220 | 0.6400 | 0.6442 | 0.4103 |
| MobileNetV3-Small (exploratory) | 1,519,906 | 5.844 | 5.917 | 10.586 | 0.6562 | 0.4889 | 0.3600 | 0.9327 | 0.4390 |

Exploratory MobileNet test accuracy: 0.8217; balanced accuracy:
0.6463; majority-class baseline: 0.8062.
Confusion matrix (actual rows, predicted columns, class order [0,1]): `[[97, 7], [16, 9]]`.

MobileNet is faster here and has fewer parameters than ResNet18, but more
parameters than the Custom CNN. Its sensitivity remains low; the table does
not establish overall superiority or statistical significance. This is one
seed and one reused test partition, with no verified patient or centre
independence and no clinical validation.

The full-precision report includes source hashes, the evaluation receipt,
original metric checks and all measured latency samples. Predictions and
checkpoints stay outside Git. Dataset-derived visualizations follow the
attribution and ShareAlike terms in `data/README.md`.
