#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Modules


# In[2]:


import os
import shutil


# In[3]:


# Define the function to organize files


# In[4]:


import os
import shutil

def organize_files(source_folder):
    count = 0
    os.chdir(source_folder)
    file_list = os.listdir()
    no_of_files = len(file_list)

    if no_of_files == 0:
        print("Error: Empty folder")
        return

    for file in file_list:
        extension = os.path.splitext(file)[1][1:].lower()
        if extension:
            dir_name = f"{extension.upper()}Files"
            new_path = os.path.join(source_folder, dir_name)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            shutil.move(file, os.path.join(new_path, file))
            count += 1

    if count == no_of_files:
        print("Operation Successful!")
    else:
        print("Failed")


# In[5]:


# Get the source folder path


# In[6]:


if __name__ == '__main__':
    source_folder = input("Enter the path of the folder to organize: ")
    organize_files(source_folder)

