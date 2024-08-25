# Todo/ Overview
We run in order
1. gen report; run inference extract samples where the model performs poorly
2. gen adversarial; generate adversarial examples + add them to the dataset + upload to hf
3. retrain

# Issues
* The data which we generate adversarially doesn't always match the input data distribution. We can fix that pretty easily by changing the way in which we prompt the llm.

# Features to Build
* Sizing. If the classifiers have lots of instances where they are wrong we should subdivide and characterize those errors further with CLIP.
* Modality. We need to generalize this approach to work with audio and image, not just text. 
* N_Samples. We need the adversarial examples which we are generating to use more examples.
* Learning Rate/ Catastrophic Forgetfulness prevention. We need to come up with a more adaptive strategy for choose the parameters which we retrain on based off of a bunch of math about the model. 
* This stuff needs to all integrate directly with hugging face. Your experimentally trained model needs to be automatically created as a separate repo. After you've done that, you need to the adversarially generated dataset needs to automatically be generated as well. The retrain script needs to pull those files in and download them, etc...