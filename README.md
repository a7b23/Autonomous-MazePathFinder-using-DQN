# MazePathFinder using deep Q learning
This program takes as input an image consisting of few blockades(denoted by block colour), the starting point denoted by blue colour and the destination denoted by green color. It outputs an image consisting of one of the possible paths from input to output. 
Shown below are the input and output of the program.

![input1](https://cloud.githubusercontent.com/assets/12389081/21818909/99b5d194-d78f-11e6-9c10-f43cdb3feffe.png)

![output1](https://cloud.githubusercontent.com/assets/12389081/21818953/bff805ca-d78f-11e6-8f65-8670cde0e9b5.png)

The input image is fed to the model consisting of 2 conv and 2 fc layers which outputs the Q values corresponding to the actions bottom and right. The agent is moved right or bottom depending on which Q value was greater and the corresponding new image generated with the new position of agent is fed again to the model.The process of getting an output state and feeding back the new image is kept on repeating untill the agent reaches the terminal stage where it reaches the destination.


#### Data Generation
The code DataGeneration.py generates the requisite data for the task. It randomly allocates the blockages of 1X1 pixel size in an image of 25X25 size .Also all the 625 images corresponding to each of the different starting positions are generated and stored in a folder.Images of 200 different such games with variations in the blockage positions are generated,thus the total training images amounting to 250 * 625 . A score associated with each of the different states(image) is also generated and stored in a txt file whereby the state where the starting point collides with blockages gets a score of -100 , when the starting point collides with destination it gets a score of 100 and all the remaining states get a score of 0.

### Training 
InitialisingTarget.py generates the inital Q values assocaitaed with each of the training images and stores them in a txt file -'Targets200_New.txt'.The generated Q values are nothing but the output of the randomly initialised model.
training2.py starts the training of the model. Randomly batch-size set of images are chosen from the trainig data and fed to the model.The model weights are updated depending on the loss which is the squared differnce between output Q-values and expected Q values. The expected Q(s,a) = r(s,a) + gamma * max(Q1(s1,a)), where max is taken over the 2 actions. Here Q1 corresponds to the q values stored in the 'Targets200_New.txt' file . Also the reward r is the difference in scores between next state and current state. 
Afetr few epochs of training the Q1 values are again updated and stored in the same txt file with the outputs being from the trained model.The new Q1 values are again used to train the model . This step of generating the target Q value and training the model is repeated for several steps untill the model learns the desired characteristics.

#### Testing
Just like DataGeneration.py , TestDataCollection.py also generates the images in the same format except that instead of 200 games it generates images corresponding to only 20 games in a seperate folder. 
testing.py takes the image corresponding to when agent is at location 0,0 for each of the 20 games and outputs the image consisiting of the final path till destination in a seperate folder . For games in which it fails to find a path because of collision with blockages no image is generated.

