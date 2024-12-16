# Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation

### This repository contains the code for our 2023 CVPR paper "Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation"

## Installation
<code>
1. Download repository <br>
2. pip install -r requirements.txt
</code>

## Attacking models
<code>
- The method relies on a taking a callable function that returns
the loss off of an adversarial image i.e. f(<strong>x_adv}</strong>). The method assumes the task is minimization problem. <br>
- View main.py for an example of how to run the method. We provide the suggested parameters there.
</code>

## Citation
<pre>
@inproceedings{williams2023black,
  title={Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation},
  author={Williams, Phoenix Neale and Li, Ke},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12291--12301},
  year={2023}
}
</pre>

## Paper
<code>
You can access the paper <a href="./CVPR_paper.pdf" download>[here]</a>.
</code>



## Install Step:
1. pip install -r /content/ga/requirements.txt
2. pip3 install autoattack@git+https://github.com/fra31/auto-attack.git@6482e4d6fbeeb51ae9585c41b16d50d14576aadc
3. pip3 install robustbench@git+https://github.com/RobustBench/robustbench.git@eecfdef49ab30b62f2e6bcff3e3fe045b054557c
4. python /content/ga/main.py