"""
CVND 06
- Thu 7 No 2019
- Marathon debugging with my technical mentor around 2 hours or so 
- Trying to fix an error in my code for Facial Keypoint detection project, but still ongoing, not solved yet
- It was great experience thought, getting help from mentor and trying different ways  and approaches, real fun
- Also learned that I need to use .forward() to predict using the saved model network
- But still not working as I wish to, need to work on more. 

"""


for (x,y,w,h) in faces:
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) 
    gray = gray.astype('float32')
    
    gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) #convert to grayscale
    gray_img = gray_img/float(255) # normalize

    roi_image = gray_img

    roi_image=cv2.resize(roi_image,(224,224)) # resize
    roi_image = roi_image.reshape(224,224,1) # reshape
    roi_image = roi_image.transpose(2,0,1) # change image input from HWC numpy shape to CHW, that of tensor

    inpt = torch.from_numpy(roi_image)
    inpt = inpt.type(torch.FloatTensor) # clarifying torch datatype
    inpt = inpt.unsqueeze(0) # unsqueeze the data to feed to model

    outp = net.forward(inpt)
    outp = outp.data.numpy() 
    outp = outp.reshape(68,2)
    pts = outp

    plt.scatter(pts[:, 0], pts[:, 1], s=20, marker='.', c='r')
    show_all_keypoints(image, pts)
