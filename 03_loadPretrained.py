"""
CVND 03
- Mon 4 Nov 2019
- 3rd ipynb of Facial Keypoint Detection
- code to load the pertained, saved model and test it in PyTorch
- Had a meeting with my tech mentor 

loading the pretrained in PyTorch
- use load_state_dict()
- BUT, you have to declare a model before hand.
- that "load_state_dict()" will be fed to the model you declare
- set ".eval()" at last



HOWEVER! My plot based on the predicted output only shows two dots! What has been go wrong! 


"""


import torch
from models import Net

net = Net() # neural net 

net.load_state_dict(torch.load('saved_models/keypoints_model_YUJIN_2_batch32epoch10`.pt'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval() # necessary to declare 'eval()' to set the environment 





image_copy = np.copy(image)

for (x,y,w,h) in faces:
    
    # Select the region of interest that is the face in the image 
    roi = image_copy[y:y+h, x:x+w]
    
    ## TODO: Convert the face region from RGB to grayscale
    gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    gray_img = gray_img.astype('float32')/255
    #norm_image = cv2.normalize(image, None, alpha=0, beta=1)
    ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    rescale_img = cv2.resize(gray_img, (224,224)) #shape 224, 224
    
    ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    img = rescale_img[np.newaxis, :]
    reshape_img = img.transpose(0,2,1)
    #print(reshape_img.shape)
    
    ## TODO: Make facial keypoint predictions using your loaded, trained network 
    
    outp = net(torch.from_numpy(reshape_img).unsqueeze(0))
    print(outp)

    ## TODO: Display each detected face and the corresponding keypoints        
    img = outp.data
    img = img.numpy()
    predicted_key_pts = img*50.0+100
    #plt.imshow(predicted_key_pts, cmap = 'gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
