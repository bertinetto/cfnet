
# End-to-end representation learning for Correlation Filter based tracking

### Work in progress! We will release the code soon.

![pipeline image][logo]

[logo]: http://www.robots.ox.ac.uk/~luca/cfnet/page1_teaser.jpg "Pipeline image"

> The Correlation Filter is an algorithm that trains a linear template to discriminate between images and their translations. It is well suited to object tracking because its formulation in the Fourier domain provides a fast solution, enabling the detector to be re-trained once per frame. Previous works that use the Correlation Filter, however, have adopted features that were either manually designed or trained for a different task. This work is the first to overcome this limitation by interpreting the Correlation Filter learner, which has a closed-form solution, as a differentiable layer in a deep neural network. This enables learning deep features that are tightly coupled to the Correlation Filter. <u>Experiments illustrate that our method has the important practical benefit of allowing lightweight architectures to achieve state-of-the-art performance at high framerates.</u>
