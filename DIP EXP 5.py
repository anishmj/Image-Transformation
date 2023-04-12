#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('city.jpeg')
plt.imshow(img)
plt.show()


# In[3]:


img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()


# In[4]:


rows,cols,dim = img.shape


# In[5]:


M = np.float32([[1,0,50],
                [0,1,75],
                [0,0,1]])


# In[6]:


translated_image = cv2.warpPerspective(img,M,(cols,rows))


# In[7]:


plt.axis('off')


# In[8]:


plt.title("Translated Image")
plt.imshow(translated_image)
plt.show()


# In[9]:


## Scaling

rows,cols,dim = img.shape
M = np.float32([[5.8,0,0],
               [0,4.5,0],
                [0,0,1]])
scaled_img = cv2.warpPerspective(img,M,(cols*2,rows*2))
plt.title("Scaled Image")
plt.imshow(scaled_img)
plt.show()


# In[10]:


## IMAGE SHEARING

M = np.float32([[1,1.5,0],
               [2.5,1,0],
               [0,0,1]])
shearing_img = cv2.warpPerspective(img,M,(int(cols*1.5),int(rows*2.5)))
plt.title("Shearing Image")
plt.imshow(shearing_img)
plt.show()


# In[22]:


M = np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
M_y = np.float32([[1,0,cols],[0,-1,rows],[0,0,1]])
ref_img = cv2.warpPerspective(img,M,(int(cols),int(rows)))
refy_img = cv2.warpPerspective(img,M_y,(int(cols),int(rows)))
plt.imshow(ref_img)
cv2.imshow('reflected_y',refy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show


# In[14]:


## image rotation

angle = np.radians(20)
M = np.float32([[np.cos(angle),-(np.sin(angle)),0],
               [np.sin(angle), np.cos(angle), 0],
               [0,0,1]])

rotated_img = cv2.warpPerspective(img,M,(int(cols),int(rows)))

plt.title("Rotation Image")
plt.imshow(rotated_img)
plt.show


# In[15]:


cropped_img = img[100:300,150:200]
plt.title("Cropped Image")
plt.imshow(cropped_img)
plt.show


# In[ ]:




