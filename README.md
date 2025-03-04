# Signal Sentinel
--- 
#### A BSc(Hons) Cyber Security dissertation project for Solent University by Josh Perryman.<br><br>This project aims to create a product that can passively detect RF jamming attacks through the use of machine learning to classify incoming signals as either safe or to be flagged up as a jamming attempt.<br><br>The idea behind this project is to create a detection system for small form embedded systems, the final product will be build on a Raspberry Pi and utalised as RTL-SDR.
---
## Why?

#### The reasoning behind why this project is planned to be used on embedded systems other than as a general radio detection system is the goal to try and protect vulnerable IoT systems from RF jamming attacks.<br>This product will be detection only allowing intergration with other systems for RF jamming mitigation.<br><br>Examples of use cases could be:

- #### IoT data loss prevention
  ##### A scernario where IoT sensors are sending constand critical data to a remote node and these messages are too valuable to be lost, if a threat actor discovers these devices and deploys a jammer to disrupt communication the sensors wont know there is an active attack and will contuine to send data that will never reach its target, on the other side a lack of communication could be caused by many reasons and in an hostile or remote envrionment these devices may not be able to be physically inspected. This product can be set up to change the data storage methid during a suspect attack, swap wireless transmission to store locally until its safe to contuine providing data intergretity 

- #### Smart cars
  ##### Car thefts have been on a constant rise, with criminals having easy access to cheap SDR devices and pre built jamming systems more hopeful crinimals are taking advange of keyless car entry and by passing cars GPS tracking devices. <br>A common method they use is to clone key fobs to gain access to the car then to deplot a jammer so the car cannot be tracked until they can remove the module and at this point recovery efforts can be lost. Pairing a passive detection system such as mine could provide an extra layer of protection for a vehicle 

  ---

  ## Current work in progress

  ##### Data collection<br>A set of safe signal data has been collected with over 300 different RF signals labaled as safe, coming from a RF Signal data set from Kaggle this will also be topped up with live capture data at a later time including common RF devices, radio commmuncation, background noise all with the idea to create a safe labaled dataset for training<br><br>The next data is the jamming labaled data, this is going to be a larger task as I could only find one dataset of jamming signals with their raw I/Q data but it lacks meta data or labels on what frequencies are being used and a lack of frequency labled in the training set could provide issues for reliable training.<br>The current workout for this is try and create simulated jamming signals through GNU radio and instead of transmitting, sent straight to a .dat file with the raw IQ data to be inputted into the program.

  ---
  ## Next steps

  #### Feature importance

  ##### Once a good size dataset has been create I can start preparing for ML training, first of all would be the data cleaning, feature importance and normalisation. The features I've chosen give a large range of statics for the models to train from but for a simple two way classification all may not be needed and could effect the overall training
 