import os 

#relative path to the repo
path='./256_ObjectCategories'
folders=os.listdir(path)
folders.sort()
for f in folders:
    print(f)

