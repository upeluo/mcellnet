# MCellNet
MCellNet is an efficient neural network for high-speed detection and classification of Cryptosporidium and Giardia from other pollutants in the water. MCellNet has an analysis speed of 346 frames per second on Jetson TX2 and accuracy >99.6%. It could be potentially applied to other high-throughput single-cell analysis applications for environmental monitoring, clinical diagnostics, and other biomedical fields.

The image database in "onetvt" that consisted of 13 classes: Cryptosporidium (2,082 images), Giardia (3,569 images), 1.54-um beads (3,466 images), 3-um beads (3,457 images) , 4-um beads (5,783 images), 4.6-um beads (2,188 images), 5-um beads (9,637 images), 5.64-um beads (3,285 images), 8-um beads (3,066 images), 10-um (8,270 images), 12-um (4,704 images), 15-um beads (2,813 images), and natural pollutants of various shapes and sizes (27,826 images). 


test-multiclass.ipynb shows the result on classifying Cryptosporidium and Giardia by using Multiclass Classifier.

test-binaryclass.ipynb shows the result on classifying protozoa class (Cryptosporidium and Giardia) by using multiclass-to-binary strategy.

training.ipynb is the script to train the network.



