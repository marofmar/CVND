'''
Haar Cascade, Face Detection
- gray scale transform, to detect face, grayscaled img is perfectly fine, since human face has diverse non-color features that the model should find out.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
- use the fully trained architecture of the face detector.
    face_casecade = cv2.CascadeClassifier('whereTheModelis.xml')
    faces = face_cascade.detectMultiScale(gray, 4, 6) 
- Input to the functions are (image, scaleFactor, minNeighbors) 
    detect more with a smaller scaleFactor & lower value of minNeighbors 
    but modifying and raising the values often result good outcome.
- Each face [x, y, length, width] 
    cv2.rectangle(img_with_detections, (x,y), (x+w, y+h), (255, 0,0), 5) 
    
    
    
AND, still struggling, the result image is not what I expect. The predicted keypoints are not located in proper place, which is the detected faces of the image.

'''

image_copy = np.copy(image) 
#print(image_copy.shape)
# 500, 759, 3 H, W, 
# 179 74 174 174
# 371 144 160 160
# loop over the detected faces from my haar cascade
"""
# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3)
    print(x,y,w,h)

fig = plt.figure(figsize=(9,9))

plt.imshow(image_with_detections)
"""
for (x,y,w,h) in faces:

    
    roi = image_copy[y:y+h, x:x+w]
    gray_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) #convert to grayscale
    #print("GRAY: ", gray_img[:5])
    gray_img = gray_img/float(255) # normalize
    #print("Gray shape:", gray_img.shape) #192 192
    #print(gray_img[:5])
    roi_image = gray_img.copy()

    roi_image= cv2.resize(roi_image,(224,224)) # resize
    roi_image = roi_image.reshape(224,224,1) # reshape
    roi_image = roi_image.transpose(2,0,1) # change image input from HWC numpy shape to CHW, that of tensor
    #roi_image = roi_image.reshape(1,1,roi_image.shape[1], roi_image.shape[2])
    inpt = torch.from_numpy(roi_image)
    inpt = inpt.type(torch.FloatTensor) # clarifying torch datatype
    inpt = inpt.unsqueeze(0) # unsqueeze the data to feed to model
#     check = inpt*255
#     print("inputs: ", check[:5])
    outp = net.forward(inpt)
    outp = outp.data.numpy()
#     outp = outp*87.0+135
    #print("right after NN: ", outp.shape) 1, 136
    x_coord = [outp[0][i] for i in range(136) if i%2==0]
    #print(x_coord)
    y_coord = [outp[0][i] for i in range(136) if i%2 ==1]

#     print("x",x_coord[:5])
#     print("y",y_coord[:5])
    outp = outp.reshape(68,2)
#     print(outp[:5])
    
    #outp = outp * float(255)
    pts = outp
    

#     plt.show()
#     for (x,y,w,h) in faces:
#     # draw a rectangle around each detected face
#     # you may also need to change the width of the rectangle drawn depending on image resolution
#         cv2.rectangle(pts,(x,y),(x+w,y+h),(255,0,0),3)
    
    
    
    
    tst = outp*w + h
    show_all_keypoints(image, tst)
    
    
    
    
    plt.figure(figsize = (3,3))

    plt.scatter(pts[:,0], pts[:,1], s = 20, marker = '.', c = 'r')
    plt.show()
    show_all_keypoints(image, pts)
    
