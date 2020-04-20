# Continuous Graph Neural Networks
Louis-Pascal A. C. Xhonneux<sup>\*</sup>, Meng Qu<sup>\*</sup>, and Jian Tang  
[https://arxiv.org/pdf/1912.00967.pdf](https://arxiv.org/pdf/1912.00967.pdf)

## Dependencies

The code has been tested under Pytorch 3.6.7 and requires the installation of the following packages and their dependencies

 - `pytorch==1.2.0`
 - `numpy==1.17.2`
 - `torchdiffeq==0.0.1`

To run the model without weights use `cd src; python run_cora_cgnn.py` and for the model with weights use `cd src; python run_cora_wcgnn.py`.

## References
If you used this code for your research, please cite this in your manuscript:

```
@misc{xhonneux2019continuous,
    title={Continuous Graph Neural Networks},
    author={Louis-Pascal A. C. Xhonneux and Meng Qu and Jian Tang},
    year={2019},
    eprint={1912.00967},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
