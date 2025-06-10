# Signal Sentinel
--- 
#### A BSc(Hons) Cyber Security dissertation project for Solent University by Josh Perryman.<br><br>This project aims to create a product that can passively detect RF jamming attacks through the use of machine learning to classify incoming signals as either safe or to be flagged up as a jamming attempt.<br><br>The idea behind this project is to create a detection system for small form embedded systems, the final product is built on a Raspberry Pi and utilises a RTL-SDR.
---

## Update 09/06/25

I've seen this project has recently started to get some attention (amazing) initially this project was finished and between this and the dissertation report I got a First class degree (amazing X2).<br>If anyone wants to try and retrain and develop a better model I'd love to see the results, open some issues or create a discussion for any questions - JP

## Why?

#### The reasoning behind why this project is planned to be used on embedded systems other than as a general radio detection system is the goal to try and protect vulnerable IoT systems from RF jamming attacks.<br>This product will be detection only allowing intergration with other systems for RF jamming mitigation.<br><br>Examples of use cases could be:

- #### IoT data loss prevention
  ##### A scernario where IoT sensors are sending constand critical data to a remote node and these messages are too valuable to be lost, if a threat actor discovers these devices and deploys a jammer to disrupt communication the sensors wont know there is an active attack and will contuine to send data that will never reach its target, on the other side a lack of communication could be caused by many reasons and in an hostile or remote envrionment these devices may not be able to be physically inspected. This product can be set up to change the data storage methid during a suspect attack, swap wireless transmission to store locally until its safe to contuine providing data intergretity 

- #### Smart cars
  ##### Car thefts have been on a constant rise, with criminals having easy access to cheap SDR devices and pre built jamming systems more hopeful crinimals are taking advange of keyless car entry and by passing cars GPS tracking devices. <br>A common method they use is to clone key fobs to gain access to the car then to deplot a jammer so the car cannot be tracked until they can remove the module and at this point recovery efforts can be lost. Pairing a passive detection system such as mine could provide an extra layer of protection for a vehicle 

  ---
 
