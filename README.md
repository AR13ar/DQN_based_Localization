# DQN_based_Localization
RL based localization and a convolutional neural network based classification algorithm. DQN-CNN is designed to enable an RL agent to learn an optimal policy for localizing the
hippocampus region and subsequently performing classification on the extracted patch. 
<p align="center">
<img src="https://github.com/AR13ar/DQN_based_Localization/assets/52975920/5cb2446a-ffaa-4585-b05e-ca76d931fe38" width="800" height="400">
</p>

The 3DModel can be replaced with a 2D model and actions in the DQN agent can be updated to work on a 2D input instead of 3D. A set of 7 actions: Up, Down, Left, Right, Top, Bottom and Terminate used for 3D model. The action Up shifts the bounding box in the image vertically by 5 units, while Down moves it downward by 5 units along the y-axis. Similarly, Left and Right actions result in the bounding box moving horizontally by 5 units to the left and right along the x-axis, respectively. The Terminate action ends the episode. The environment looks like the following.
<p align="center">
<img src= "https://github.com/AR13ar/DQN_based_Localization/assets/52975920/82b7493d-7ead-4d9d-90e1-b6a758a4b555" width="400" height="300">
</p>

Some acceptable and unacceptable results for the 2D localization is shown below. The classification accuracy of the model was compared with DL models trained on Ground truth Masks. This method achieved F1-score within error of 3.7% and 1.1% and accuracy within error of 6.6% and 1.6% with the supervised models while achieving the highest recall score.

Acceptable Localization 
<p align="center">
<img src="https://github.com/AR13ar/DQN_based_Localization/assets/52975920/764f3e61-c442-45e9-98e7-2273d79dcadd" width="300" height="200">
</p>
Unacceptable Localization
<p align="center">
<img src="https://github.com/AR13ar/DQN_based_Localization/assets/52975920/3c3a9c38-d392-4ecc-aa7e-36e7dd861d74" width="300" height="200">
</p>
