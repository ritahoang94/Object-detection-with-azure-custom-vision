# Plant detection with Azure Custom vision

_The model is trained with customvision.ai_

_Author: Rita Hoang_

_Ask me at: ngochoanb@gmail.com_

## 1. Training:
### 1.1 Prepare Azure resource
- Create Azure custom vision resources: Go to https://www.customvision.ai/ 
- Sign in to your Azure Account
- Create custom vision resources https://portal.azure.com/?microsoft_azure_marketplace_ItemHideKey=microsoft_azure_cognitiveservices_customvision#create/Microsoft.CognitiveServicesCustomVision
- Create new project: New project > enter name and description > Resource (pick the resource just created) > Domain: Pick General (compact) > Click on Create Project

Reference: https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/get-started-build-detector


### 1.2 Prepare input for training

To create input for training, I filmed 2 videos of my pileadepressa pot and rubberplant pot. Then, I will use a python script to extract frames in the video and save it as images. 

To make sure that I won't save exactly the same images, I only save one in every 100 images in the video.

Run the following code to extract images to input/images folder.


```
python3 src/extract_frame.py  pileadepressa
python3 src/extract_frame.py  rubberplant
```

### 1.3 Annotation
On Custom Vision portal, go to the project you just created > Click on add images > upload images in input/images folder > mannually annotate each image. (Custom vision will automatically suggest the bounding boxes around the objects to help accelerate the mannual annotation process).

### 1.4 Train & Review model result
- On the top toolbar, click on train > Choose quick training (You can choose advanced training if you want a better result)
- Once the training is completed, go to Performance on the toolbar to review the model performance (Precision, recall, mAP)
- You can also go to quick test and upload a few images to test the model result
- If you aren't happy with the result, add more images to the training set and repeat step 1.3 & 1.4

## 2. Inference
Now, you already have a good  model, it's time to use it for our applications. Let's say, we can use this model to do object detection for streaming or recoded videos.

- In Performance tab of Custom Vision portal, click on export model
- Export as TensorFlow format
- Download the artifacts (there will be a zipfile, which contains the model and some boilderplat code)
- I prepared some test video, to test the model, run the following code

in terminal 
```
export PYTHONPATH=[PATH_TO_THIS_DIRECTORY]
```

Run inference - test 1: test with a different video background; test 2: test with the same video background as training videos
```
python3 src/video_predict.py  [test number]
```


Here is the final result when apply the model on live video:

https://user-images.githubusercontent.com/42925756/168947640-178fed92-d528-4fef-858d-8c2b84d69766.mp4


For sample output result, check folder __output__




