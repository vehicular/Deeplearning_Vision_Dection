
## Deep Learning Object Detection Model

+ [Tensorflow API](https://github.com/tensorflow/models/tree/master/research/object_detection)

  1. Transform xml file to cvs format
  
  2. Create TF Record which can be used by the model directly
  
  3. Download a [Base Model](https://github.com/bourdakos1/Custom-Object-Detection/blob/master/object_detection/g3doc/detection_model_zoo.md) 
  
  4. Train the Model and use [Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) to visualize the process

     ![alt text](https://github.com/vehicularai/Jay_Deng/blob/master/Images/Tensorboard.png "Logo Title Text 1")

      
  
  5. Export the Inference Graph
  
  6. Test the Model
  
     ![alt text](https://github.com/vehicularai/Jay_Deng/blob/master/Images/Infrared-Detection.png "Logo Title Text 1")
 
+ [IBM PowerAI](https://github.com/IBM/powerai-vision-object-detection)

  Powerful tool and very easy to use but currently only for pictures rather than real-time videos
  
+ [Easy YOLO](https://github.com/llSourcell/YOLO_Object_Detection)
  
  This is a good platform as well, it's as simple as Tensorflow API, it can be highly costomized to users based on their own dataset.

## Simulation Source Collection

 +  [Apollo](https://github.com/ApolloAuto/apollo) has full access of various aspects of self-driving system. Including Localization, Perception, Planning, Prediction etc.
 ![alt text](https://github.com/vehicularai/Jay_Deng/blob/master/Images/Apollo.png "Logo Title Text 1")
     
    The problem is the platform is open, but the code is not. It's not easy to use the algrithm from them but can use the module from them.
