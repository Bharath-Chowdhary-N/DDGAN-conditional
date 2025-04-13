import cv2
from astropy.io import fits
import os
import numpy as np

def convert_shape(npy_array):
    
    dim = (101,101)
    
    return cv2.resize(npy_array, dim, interpolation = cv2.INTER_AREA)


def normalize_img(numpy_img):
    numpy_img = ((numpy_img - np.amin(numpy_img)) / (np.amax(numpy_img) - np.amin(numpy_img)))
    return numpy_img


def norm_and_sqrt(data_batch):
    
    norm_data_batch = normalize_img(data_batch)
    
    sqrt_data_batch = np.sqrt(norm_data_batch)
    
    return sqrt_data_batch

def convert_jpg_as_fits(export_array,global_ite):
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())

    for img in export_array:
        hdul.append(fits.ImageHDU(data=img))
        folder_name = "fits_folder/" + str(global_ite) + "/"
        os.makedirs(folder_name)
        file_name = str(global_ite) + ".fits"
        full_name = folder_name + file_name
        hdul.writeto(full_name)
        global_ite+=1
        #print("{} ite completed".format(global_ite))
    return global_ite
        
    
def save_npy_as_jpg(file,global_ite):
    
    #resized_npy_array = convert_shape(npy_array)
    
    
        
    full_file_path = "temp_npy_files/" + file

    npy_array = np.load(full_file_path)

    image = [convert_shape(i) for i in npy_array]

    data_batch = np.stack(image, axis=0)

    data_batch_1d = data_batch[:,:,:,0]

    data_batch_1d_reshaped = np.reshape(data_batch_1d,[184,101,101,1])

    data_batch_norm_sqrt = norm_and_sqrt(data_batch_1d_reshaped)

    global_ite = convert_jpg_as_fits(data_batch_norm_sqrt,global_ite)

    print("-------------------------file: {} has been processed--------------------------------".format(file))

    return global_ite


if __name__ == "__main__":
        file_name_list = os.listdir("temp_npy_files/")
        print("Num files are : ",len(file_name_list))
        global_ite=0
        for file in file_name_list:
             global_ite = save_npy_as_jpg(file, global_ite)
