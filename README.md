
# End-to-end representation learning for Correlation Filter based tracking

### Work in progress! We will release the code soon.

![pipeline image][logo]

[logo]: http://www.robots.ox.ac.uk/~luca/cfnet/page1_teaser.jpg "Pipeline image"

- - - -
> The Correlation Filter is an algorithm that trains a linear template to discriminate between images and their translations. It is well suited to object tracking because its formulation in the Fourier domain provides a fast solution, enabling the detector to be re-trained once per frame. Previous works that use the Correlation Filter, however, have adopted features that were either manually designed or trained for a different task. This work is the first to overcome this limitation by interpreting the Correlation Filter learner, which has a closed-form solution, as a differentiable layer in a deep neural network. This enables learning deep features that are tightly coupled to the Correlation Filter. <u>Experiments illustrate that our method has the important practical benefit of allowing lightweight architectures to achieve state-of-the-art performance at high framerates.</u>
- - - -

#### Getting started

[ **Tracking only** ] If you don't care about training, you can simply use one of our pretrained networks with our basic tracker.
  1. Prerequisites: GPU, CUDA (we used 7.5), [cuDNN](https://developer.nvidia.com/cudnn) (we used v5.1), Matlab, [MatConvNet](http://www.vlfeat.org/matconvnet/install/).
  2. Clone the repository.
  3. Download the pretrained networks from (here)[https://bit.ly/cfnet_networks] and unzip the archive in `cfnet/pretrained`.
  4. Go to `cfnet/src/tracking/` and remove the trailing `.example` from `env_paths_tracking.m.example`, `startup.m.example`, editing the files as appropriate.
  5. Be sure to have at least one video sequence in the appropriate format. The easiest thing to do is to download the validation set (from [here](https://bit.ly/cfnet_validation)) that we used for tracking evaluation and then extract the `validation` folder in `cfnet/data/`.
  6. `cfnet/src/tracking/run_tracker_evaluation.m` is the entry point. Default hyper-params are at the start of `tracker.m` and are overwritten by custom ones specified in `run_tracker_evaluation.m`.

 [ **Training and tracking** ] If you prefer to DIY and train your own network, the process is slightly more involved (but also more fun).
  1. Prerequisites: GPU, CUDA (we used 7.5), [cuDNN](https://developer.nvidia.com/cudnn) (we used v5.1), Matlab, [MatConvNet](http://www.vlfeat.org/matconvnet/install/).
  2. Clone the repository.
  3. Follow these [step-by-step instructions](https://github.com/bertinetto/siamese-fc/tree/master/ILSVRC15-curation), which will help you generating a curated dataset compatible with the rest of the code.  
  4. If you did not generate your own metadata (point `7.` of the instructions), download [imdb_video_2016-10.mat](bit.ly/cfnet_imdb_video) (6.7GB) with all the metadata and the [dataset stats](http://bit.ly/imdb_video_stats). Put them in `cfnet/data/`.
  5. Go to `cfnet/src/training` and remove the trailing `.example` from `env_paths_training.m.example` and `startup.m.example`, editing the files as appropriate.
  6. The various `cfnet/train/run_experiment_*.m` are some examples to start training. Default hyper-params are at the start of `experiment.m` and can be overwritten by custom ones specified in `run_experiment_*.m`.
  7. By default, training plots are saved in `cfnet/src/training/data/`. When you are happy, grab a network snapshot (`net-epoch-X.mat`) and save it somewhere (e.g. `cfnet/pretrained/`).
  8. Go to point `4.` of <i>Tracking only</i>, follow the instructions and enjoy the labour of your own GPUs!