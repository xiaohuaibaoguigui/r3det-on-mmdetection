from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
import numpy as np
import transforms
import onnxruntime
from torch.autograd import Variable
from onnxruntime.datasets import get_example
def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    print("result is ",len(result))
    for i in range(len(result)):
        print("result ",i," is ",len(result[i]))
    print("get result!!!!!!!!!!!!!",result)
    # show the results
    #show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    
def testonnx(onnxmodelpath,dummy_input):
    example_model = get_example(onnxmodelpath)
    session = onnxruntime.InferenceSession(example_model)
    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name  
    print('Input Name:', input_name)
    result = session.run([], {input_name: dummy_input.data.numpy()})
    
    return result
def load_image1(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    image=cv2.resize(image,(540,960))
    #image = np.dstack((image, np.fliplr(image)))
    #image = image.transpose((2, 0, 1))
    #image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = Variable(image)
    return image
def testByONNX():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('onnxpath', help='onnx file path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    dummy_input=load_image1(args.img) 
    result=testonnx(args.onnxpath,dummy_input)
    print("result shape is ",len(result[0][0]))
    for i in range(len(result)):
        print("result ",i," is ",len(result[i][0][0]))
    print("the result is {}".format(result[0][0]))
if __name__ == '__main__':
    #main()
    testByONNX()
    
