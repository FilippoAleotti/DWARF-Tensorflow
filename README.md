# Dwarf-Tensorflow
TensorFlow implementation of `Learning end-to-end scene flow by distilling single tasks knowledge`.

[Paper](https://arxiv.org/pdf/1911.10090.pdf), [Video](https://www.youtube.com/watch?v=qGWpi3z2M74&t=62s)

![](assets/banner.gif)
## Abstract
Scene flow is a challenging task aimed at jointly estimating the 3D structure and motion of the sensed environment.
Although deep learning solutions achieve outstanding performance in terms of accuracy, these approaches divide the
whole problem into standalone tasks (stereo and optical flow)
addressing them with independent networks. Such a strategy dramatically increases the complexity of the training
procedure and requires power-hungry GPUs to infer scene
flow barely at 1 FPS. Conversely, we propose DWARF, a
novel and lightweight architecture able to infer full scene
flow jointly reasoning about depth and optical flow easily
and elegantly trainable end-to-end from scratch. Moreover,
since ground truth images for full scene flow are scarce,
we propose to leverage on the knowledge learned by networks specialized in stereo or flow, for which much more
data are available, to distill proxy annotations. Exhaustive
experiments show that i) DWARF runs at about 10 FPS
on a single high-end GPU and about 1 FPS on NVIDIA
Jetson TX2 embedded at KITTI resolution, with moderate drop in accuracy compared to 10× deeper models, ii)
learning from many distilled samples is more effective than
from the few, annotated ones available.

## Requirements
The code has been tested with python3 and Tensorflow 1.10.
Dependencies may be installed running  

```Bash
pip install -r requirements.txt
```

In order to use the [KITTI evaluation suite](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php), you need to install:

```Bash
sudo apt-get install libpng++-dev
sudo apt-get install libpng-dev
```
**Note**: gcc and g++ version should have version 5

## Network 

![dwarf](./assets/dwarf.png?raw=true "Title")

## Filenames
* `kitti_160`: split used for training the Gt model in the ablation
* `kitti_mixed`: split used for training the Gt+Px model in the ablation
* `kitti_multiview_partial`: split used for training the Px model in the ablation
* `kitti_multiview`: split used for first training on PX for the final submission
* `kitti_200`: split used for final fine tuning on Gt for the final submission

## Pretrained models

Models are available for download

| Training  | Dataset  | zip  |
|:-:|:--:|:--:|
| Synthetic pretraining | Flying Things 3D  |  [weights](https://drive.google.com/file/d/1zsr01odgY4DI4LbXrlcyB8al395MFQsu/view?usp=sharing) |
| Final submission | KITTI 2015  |  [weights](https://drive.google.com/open?id=1b-1OhhKjQVW3QYmqjdhT5LGYNNuYbjJq) |

## Proxy labels generation

Proxy labels have been generated using the network proposed by Ilg et al. in [Occlusions, Motion and Depth Boundaries with
a Generic Network for Disparity, Optical Flow
or Scene Flow Estimation](https://arxiv.org/pdf/1808.01838.pdf).
**WE DO NOT** make them available for download. You can generate these proxy by your own running their code, available [here](https://github.com/lmb-freiburg/netdef_models).

Consider to cite them when using their work:

```
@inproceedings{ilg2018occlusions,
  title={Occlusions, motion and depth boundaries with a generic network for disparity, optical flow or scene flow estimation},
  author={Ilg, Eddy and Saikia, Tonmoy and Keuper, Margret and Brox, Thomas},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={614--630},
  year={2018}
}
```
## Times

We took inference time using CUDA layers for 1D, 2D and 3D correlations. However, we notice that these layers are unstable and do not guarantee the same results for consecutive inferences, so we suggest to use tensorflow implementations for testing.

## Reference:
If you find the code useful, please cite our paper:

```
@inproceedings{aleotti2020learning,
  title={Learning End-To-End Scene Flow by Distilling Single Tasks Knowledge},
  author={Aleotti, Filippo and Poggi, Matteo and Tosi, Fabio and Mattoccia, Stefano},
  booktitle={Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}
```