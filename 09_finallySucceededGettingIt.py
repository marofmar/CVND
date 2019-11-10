"""
Mon 11 Nov 2019
Facial Keypoint Detection on two faces in an image.
- Drew a subplot so that the result is explicitly shown 
- Added padding to the interested region of the image, since the trained model that I am using here has mastered to find the key points out of somewhat sparse image containing one face out of it.
- Super happy that I finally made it.
- I found out that the desired outcome is not putting the points on the whole image, but the only face image which is cropped from the original one. This was way easier than the version I misunderstood.


"""

image_copy = np.copy(image) 
#print(image_copy.shape)
# 500, 759, 3 H, W, 
# 179 74 174 174
# 371 144 160 160
# loop over the detected faces from my haar cascade

plt.figure(figsize = (18,6))
for i, (x,y,w,h) in enumerate(faces):


    ax = plt.subplot(1,2,i+1)
    #### padding
#     x_pad = int(w*0.5)
#     y_pad = int(h*0.5)
#     roi = image_copy[y-y_pad:y+y_pad, x-x_pad:x+x_pad]

    #### non padding
    roi = image_copy[y:y+int(1.2*h), x:x+int(1.2*w)] # seems like I cannot put padding on the starting part of the array (negative value prob maybe)
    gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) #convert to grayscale
    #print("GRAY: ", gray_img[:5])
    gray_img = gray_img/255.0 # normalize
    print(gray_img.min(), gray_img.max())
    #print("Gray shape:", gray_img.shape) #192 192
    #print(gray_img[:5])
    roi_image = gray_img.copy()

    roi_image= cv2.resize(roi_image,(224,224)) # resize
    
    
    roi_image = roi_image.reshape(1,1,224,224) # reshape, # change image input from HWC numpy shape to CHW, that of tensor
    #roi_image = roi_image.transpose(2,0,1)
    #roi_image = roi_image.reshape(1,1,roi_image.shape[1], roi_image.shape[2])
    inpt = torch.from_numpy(roi_image)
    inpt = inpt.type(torch.FloatTensor) # clarifying torch datatype
    #inpt = inpt.unsqueeze(0) # unsqueeze the data to feed to model
#     check = inpt*255
#     print("inputs: ", check[:5])
    outp = net.forward(inpt)
    torch.squeeze(outp)
    outp = outp.data.numpy()*50+87
#     outp = outp*87.0+135
#     #print("right after NN: ", outp.shape) 1, 136
#     x_coord = [outp[0][i] for i in range(136) if i%2==0]
#     #print(x_coord)
#     y_coord = [outp[0][i] for i in range(136) if i%2 ==1]

#     print("x",x_coord[:5])
#     print("y",y_coord[:5])
    outp = outp.reshape(68,2)
#     print(outp[:5])
    
    #outp = outp * float(255)
    pts = outp
    plt.imshow(gray_img, cmap='gray')
    plt.scatter(pts[:, 0], pts[:, 1], s=40, marker='.', c='b')
plt.show()
