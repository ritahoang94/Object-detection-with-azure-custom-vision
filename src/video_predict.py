import sys
import tensorflow as tf
import numpy as np
from PIL import Image
import azure_model.python.predict as predict
import cv2

MODEL_FILENAME = 'azure_model/model.pb'
LABELS_FILENAME = 'azure_model/labels.txt'


def main(test_number):
    # Load a TensorFlow model
    video = "input/test/test" + test_number + ".MOV"
    output_path = "output/output" + test_number + ".mp4"


    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = predict.TFObjectDetection(graph_def, labels)


    vidcap = cv2.VideoCapture(video)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, 5, size)

    count = 0
    while(vidcap.isOpened()):

        success,image = vidcap.read()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        
        
        if success == True:
            count += 1
            if count % 3 ==0:
                predictions = od_model.predict_image(im_pil)
                print(len(predictions))
                width, height = im_pil.size
                im_np = np.asarray(im_pil)
                im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)

                for each_pred in predictions:
                    bounding_box = each_pred["boundingBox"]
                    left = bounding_box['left'] * width
                    top = bounding_box['top'] * height
                    w = bounding_box['width'] * width
                    h = bounding_box['height'] * height

                    im_np = cv2.rectangle(im_np, (int(left),int(top)), (int(left+w), int(top + h)), (255, 255, 255), 2)
                    im_np = cv2.putText(im_np, str(each_pred['tagName']) + "-" + str(round(each_pred['probability'],2)), (int(left),int(top)), cv2.FONT_HERSHEY_SIMPLEX,  2, (255, 255, 255), 2, cv2.LINE_AA)
                    

                cv2.imshow("image",im_np)
                out.write(im_np)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        else:
            break

    out.release()
    vidcap.release()
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} image_filename'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
