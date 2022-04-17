import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import transforms as T

def showbbox(model, img, device):
    # The img entered is a tensor in the 0-1 range        
    model.eval()
    with torch.no_grad():
        '''
        prediction Like:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])
        
    print(prediction)
        
    img = img.permute(1,2,0)  # C,H,W_H,W,C, for drawing
    img = (img * 255).byte().data.cpu()  # * 255, float to 0-255
    img = np.array(img)  # tensor → ndarray
    
    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())
        
        label = prediction[0]['labels'][i].item()
        
        # if label == 1:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
        #Crop image
        cropped_image = img[xmin:xmax, ymin:ymax]
    #TODO : cropped_image, label as atuple/different dataset.

    
    plt.figure(figsize=(20,15))
    plt.imshow(img)


def get_cropped_box(img, bb_coords):
    cropped_images = []
    for xmin, ymin, xmax, ymax in bb_coords:
        torch.Tensor(img[xmin:xmax, ymin:ymax])
        cropped_images.append(img[xmin:xmax, ymin:ymax])
    #TODO : cropped_image, label as atuple/different dataset.

    return cropped_images
    

def get_transform():
    #AMke it 
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(transforms.Resize(224))
    return transforms