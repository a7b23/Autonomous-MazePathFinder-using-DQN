# MazePathFinder using deep Q learning
This program takes as input an image consisting of few blockades(denoted by block colour), the starting point denoted by blue colour and the destination denoted by green color. It outputs an image consisting of one of the possible paths from input to output. 
Shown below are the input and output of the program.

![input1](https://cloud.githubusercontent.com/assets/12389081/21818909/99b5d194-d78f-11e6-9c10-f43cdb3feffe.png)

![output1](https://cloud.githubusercontent.com/assets/12389081/21818953/bff805ca-d78f-11e6-8f65-8670cde0e9b5.png)

The input image is fed to the model consisting of 2 conv and 2 fc layers which outputs the Q values corresponding to the actions bottom and right. The agent is moved right or bottom depending on which Q value was greater and the corresponding new image generated with the new position of agent is fed again to the model.The process of getting an output state and feeding back the new image is kept on repeating untill the agent reaches the terminal stage where it reaches the destination.


