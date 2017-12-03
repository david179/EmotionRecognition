# How to run

Use openface to get the AUs.

Get name of docker container 

`docker ps`

Copy images to docker Openface

`docker cp <folder_to_copy>  <docker_container_name>:/opt/OpenFace/build/bin/images`

`./FaceLandmarkImg -fdir /path_to_images -ofir /path_to_output_directory`

Use parse.py to extract the AU intensities and occurences from the docker output files and save this data in .npy files

`python parse.py`

Use train.py to train the model and predict

To train

`python train.py --train=yes`

To predict put your prediction images in the folder predict_images and their AUs in predict_images_AU

`python train.py -e=yes`