"""
#30DaysofUdacity #CVND #PractiveMakesPerfect I practiced implementing codes for image processing.
- I took a lecture of YOLO. I heard of the impressive name, but never took time to seriously figure out what this is. So, basically this was the first time ever studying this subject. 
- Also tried to implement denormalizing codes for 3rd notebook in Facial Key points Detection project. I only multiplied 225 to the output, but maybe should use the (x,y,w,h) from the Haar Cascades output
- I watched a Andrew Ng’s Machine Learning  video, Lecture1. I should keep watching this great series one each day. 

"""

#####subplot

image_copy = np.copy(image)

# loop over the detected faces from my haar cascade
for (x,y,w,h) in faces:

#     plt.figure(figsize = (10,10))


#     ax = plt.subplot(2,1,i+1)
#     i+=1
    roi = image_copy[y:y+h, x:x+w]
    width = roi[1]
    height = roi[0]
    roi_g = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) 
    roi_n = roi_g.astype('float32')/float(255)
    
    plt.imshow(roi_n)
    roi_image = roi_n.copy()

    roi_image = cv2.resize(roi_image,(224,224)) # resize
    roi_image = roi_image.reshape(224,224,1) # reshape
    roi_image = roi_image.transpose(2,0,1) # change image input from HWC numpy shape to CHW, that of tensor

    inpt = torch.from_numpy(roi_image)
    inpt = inpt.type(torch.FloatTensor) # clarifying torch datatype
    inpt = inpt.unsqueeze(0) # unsqueeze the data to feed to model
    
    outp = net(inpt)
    outp = outp.view(68, -1)
    print("NN output: ",outp.shape)
    
    outp = outp.data
    outp = outp.numpy()
    outp = outp*60.0 +96
    
    
#     #outp = net.forward(inpt)
#     outp = outp.data.numpy() 
#     outp = outp.reshape(68,2)
#     #outp = outp * float(255)
    
    pts = outp

    #plt.scatter(pts[:, 0], pts[:, 1], s=20, marker='.', c='r')
    plt.figure(figsize = (10,10))
    plt.imshow(image, cmap = 'gray') 
    plt.scatter(pts[:,0], pts[:,1], s = 20, marker = '.', c = 'r')
    #show_all_keypoints(image, pts)
plt.show()
