import os 

#relative path to the repo
path='./256_ObjectCategories'
folders=os.listdir(path)
folders.sort()
for f in folders:
    print(f)


#read the folders and keep images in different categories

#data pre-processing
# make the image square
# flatten

