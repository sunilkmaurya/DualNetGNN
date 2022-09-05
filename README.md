# Dual-Net GNN
Implementation of Dual-Net GNN.
Experiments were conducted with following setup:  
Pytorch: 1.6.0  
Python: 3.8.5  
Cuda: 10.2.89
Trained on NVIDIA V100 GPU.

Main Experiment Settings:

Hidden dimension: 64

No. of hops: 3 ( Total 7 feature matrices )

p : 4 ( up to 4 feature matrices can be selected )


**Summary of results**

| **Dataset** | **Accuracy (%)** | **Avg. train time, 3-runs (sec)** |
| :---------- | :---------------: | :--------------------------------: |
| Cora        | 87\.77           | 36\.53                             |
| Citeseer    | 77\.15           | 46\.64                             |
| Pubmed      | 89\.64           | 71\.58                             |
| Chameleon   | 78\.46           | 59\.95                             |
| Wisconsin   | 89\.41           | 19\.45                             |
| Texas       | 87\.57           | 46\.83                             |
| Cornell     | 88\.11           | 44\.84                             |
| Squirrel    | 73\.97           | 88\.55                            |
| Actor       | 37\.29           | 52\.13                             |

**Run node classification:**

```./run_classification.sh```




(Results may vary slightly with a different platform, e.g. use of different GPU. In such case, for best performance, some hyperparameter search may be required. Please refer to [PyTorch documentation](https://pytorch.org/docs/stable/notes/randomness.html) for more details.)

Datasets and parts of preprocessing code were taken from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn) and [GCNII](https://github.com/chennnM/GCNII) repositories. We thank the authors of these papers for sharing their code.


