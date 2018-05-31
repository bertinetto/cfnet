
# End-to-end representation learning for Correlation Filter based tracking

![pipeline image][logo]

[logo]: http://www.robots.ox.ac.uk/~luca/cfnet/page1_teaser.jpg "Pipeline image"

- - - -
Project page: [http://www.robots.ox.ac.uk/~luca/cfnet.html]
- - - -
**WARNING**: we used Matlab 2015, MatConvNet v1.0beta24, CUDA 8.0 and cudnn 5.1. Other configurations might work, but it is not guaranteed. In particular, we received several reports of problems with Matlab 2017.
- - - -

#### Getting started

[ **Tracking only** ] If you don't care about training, you can simply use one of our pretrained networks with our basic tracker.
  1. Prerequisites: GPU, CUDA (we used 7.5), [cuDNN](https://developer.nvidia.com/cudnn) (we used v5.1), Matlab, [MatConvNet](http://www.vlfeat.org/matconvnet/install/).
  2. Clone the repository.
  3. Download the pretrained networks from [here](https://drive.google.com/open?id=0B7Awq_aAemXQZ3JTc2l6TTZlQVE) and unzip the archive in `cfnet/pretrained`.
  4. Go to `cfnet/src/tracking/` and remove the trailing `.example` from `env_paths_tracking.m.example`, `startup.m.example`, editing the files as appropriate.
  5. Be sure to have at least one video sequence in the appropriate format. The easiest thing to do is to download the validation set (from [here](https://drive.google.com/file/d/0B7Awq_aAemXQSnhBVW5LNmNvUU0/view?usp=sharing)) that we used for the tracking evaluation and then extract the `validation` folder in `cfnet/data/`.
  6. Start from one of the `cfnet/src/tracking/run_*_evaluation.m` entry points.

 [ **Training and tracking** ] Start here if instead you prefer to DIY and train your own networks.
  1. Prerequisites: GPU, CUDA (we used 7.5), [cuDNN](https://developer.nvidia.com/cudnn) (we used v5.1), Matlab, [MatConvNet](http://www.vlfeat.org/matconvnet/install/).
  2. Clone the repository.
  3. Follow these [step-by-step instructions](https://github.com/bertinetto/siamese-fc/tree/master/ILSVRC15-curation), which will help you generating a curated dataset compatible with the rest of the code.  
  4. If you did not generate your own metadata, download [imdb_video_2016-10.mat](https://drive.google.com/file/d/0B7Awq_aAemXQMFpSUU90OW5oaXc/view?usp=sharing) (6.7GB) with all the metadata and also the [dataset stats](https://drive.google.com/file/d/0B7Awq_aAemXQcndzY3M5dkprVTA/view?usp=sharing). Put them in `cfnet/data/`.
  5. Go to `cfnet/src/training` and remove the trailing `.example` from `env_paths_training.m.example` and `startup.m.example`, editing the files as appropriate.
  6. The various `cfnet/train/run_experiment_*.m` are some examples to start training. Default hyper-params are at the start of `experiment.m` and are overwritten by custom ones specified in `run_experiment_*.m`.
  7. By default, training plots are saved in `cfnet/src/training/data/`. When you are happy, grab a network snapshot (`net-epoch-X.mat`) and save it somewhere (e.g. `cfnet/pretrained/`).
  8. Go to point `4.` of <i>Tracking only</i>, follow the instructions and enjoy the labour of your own GPUs!
