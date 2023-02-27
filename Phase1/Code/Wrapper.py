#!/usr/bin/env python

"""
CMSC733 Spring 2021: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Sakshi Kakde
M.Eng. Robotics,University of Maryland, College Park
"""

# Code starts here:
import sys
from pathlib import Path
sys.path.append(str(Path( __file__ ).parent.joinpath('..')))
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import imutils
import os
import sklearn.cluster
import os

def loadImages(folder_name, files):
	print("Loading images from ", folder_name)
	images = []
	if files == None:
		files = os.listdir(folder_name)
	print(files)
	for file in files:
		image_path = folder_name + "/" + file
		image = cv2.imread(image_path)
		if image is not None:
			images.append(image)
			
		else:
			print("Error in loading image ", image)

	return images

def jpg2pngList(jpg_list):
	png_list = []
	for n in range(len(jpg_list)):
		s = jpg_list[n]
		f_name = str()
		for i in range(len(s)):
			if s[i] == '.':
				break

			f_name += str(s[i])	

		png_list.append(str(f_name) + ".png")
	
	return png_list


def halfDisk(radius, angle):
	size = 2*radius + 1
	centre = radius
	half_disk = np.zeros([size, size])
	for i in range(radius):
		for j in range(size):
			distance = np.square(i-centre) + np.square(j-centre)
			if distance <= np.square(radius):
				half_disk[i,j] = 1
    
	
	half_disk = imutils.rotate(half_disk, angle)
	half_disk[half_disk<=0.5] = 0
	half_disk[half_disk>0.5] = 1
	return half_disk

def convolve2d(input, kernal):
    #output size same as input size, stride = 1
	
	kernal_size = kernal.shape[0]
	#input_size = input.shape[0]
	input_width = input.shape[0]
	input_height = input.shape[1]
	out = np.zeros([input_width, input_height])
	padding = int((kernal_size - 1)/2)
	input_padded = np.pad(input, ((padding, padding), (padding, padding)), 'constant')
	for w in range(input_width):
		for h in range(input_height):
			ip = input_padded[w:w + kernal_size, h:h + kernal_size]
			out[w,h] =  np.sum(ip * kernal)

	return out

def gaussian2d_rotated(sigma, theta, size):
	sigma_x, sigma_y = sigma
	gaussian = np.zeros([size, size])

	if (size%2) == 0:
		index = size/2
	else:
		index = (size - 1)/2

	x, y = np.meshgrid(np.linspace(-index, index, size), np.linspace(-index, index, size))
	points = [x.flatten(), y.flatten()]
	points = np.array(points)
	points = points.transpose()
	rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
	points_rotated = np.dot(points, rotation_matrix)
	x_rotated = points_rotated[:,0].reshape([size, size])
	y_rotated = points_rotated[:,1].reshape([size, size])
	pow = (np.square(x_rotated)/np.square(sigma_x)) + (np.square(y_rotated)/np.square(sigma_y))
	pow /= 2
	gaussian = (0.5/(np.pi * sigma_x * sigma_y)) * np.exp(-pow)
	return gaussian


def gaussian2d(sigma, size):	
	sigma_x, sigma_y = sigma
	gaussian = np.zeros([size, size])
	if (size%2) == 0:
		index = size/2
	else:
		index = (size - 1)/2

	x, y = np.meshgrid(np.linspace(-index, index, size), np.linspace(-index, index, size))
	pow = (np.square(x)/np.square(sigma_x)) + (np.square(y)/np.square(sigma_y))
	pow /= 2
	gaussian = (0.5/(np.pi * sigma_x * sigma_y)) * np.exp(-pow)
	return gaussian

def sin2d(frequency, size, angle):
	if (size%2) == 0:
		index = size/2
	else:
		index = (size - 1)/2

	x, y = np.meshgrid(np.linspace(-index, index, size), np.linspace(-index, index, size))
	mu = x * np.cos(angle) + y * np.sin(angle)
	sin2d = np.sin(mu * 2 * np.pi * frequency/size)

	return sin2d

def printFilterbank(filter_bank, file_name, cols):
	#cols = 6
	rows = math.ceil(len(filter_bank)/cols)
	plt.subplots(rows, cols, figsize=(15,15))
	for index in range(len(filter_bank)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(filter_bank[index], cmap='gray')
	
	plt.savefig(file_name)
	plt.close()

# function to genarate filters with o orientation, s scales and filter size as n*n
# Generate gaussian with size = filter_size
# add padding to retain the size
# Convolve with sobel x and sobel y to get Gx and Gy
# DoG = Gx*cos theta + Gy*sin theta
def DoGFilters(orientations, scales, filter_size):

	filter_bank = []
	#sobel kernals
	Sx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	Sy = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])
	#print("Sobel kernals are:\nGx =", Sx, "\nGy = ", Sy)  
    
	
	for scale in scales:
		sigma = [scale, scale]
		G = gaussian2d(sigma, filter_size)
		#print("The gaussian matrix is:", G)
		#Gx = convolve2d(G, Sx)
		#Gy = convolve2d(G, Sy)
		Gx = cv2.filter2D(G,-1, Sx)
		Gy = cv2.filter2D(G,-1, Sy)
		for orientation in range(orientations):
			filter_orientation = orientation * 2 * np.pi / orientations 
			filter = (Gx * np.cos(filter_orientation)) +  (Gy * np.sin(filter_orientation))
			filter_bank.append(filter)
        
		
	return filter_bank

def LMFilter(scales, orientations, filter_size):
	scale_1 = scales[0:3]
	gaussian_scale = scales
	log_scale = scales + [i * 3 for i in scales]

	filter_bank = []
	first_derivatives = []
	second_derivatives = []
	gaussian = []
	LoG = []
	#del_x = np.array([[-1, 1],[-1, 1]])
	#del_y = np.array([[-1, -1],[1, 1]])
	del_x = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
	del_y = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]])

	for scale in scale_1:
		sigma = [3*scale, scale]
		G = gaussian2d(sigma, filter_size)
		#first_derivative = np.sqrt(np.square(convolve2d(G, del_x)) + np.square(convolve2d(G, del_y)))
		#second_derivative = np.sqrt(np.square(convolve2d(first_derivative, del_x)) + np.square(convolve2d(first_derivative, del_y)))
		#first_derivative = convolve2d(G, del_x) + convolve2d(G, del_y)
		#second_derivative = convolve2d(first_derivative, del_x) + convolve2d(first_derivative, del_y)

		first_derivative = cv2.filter2D(G, -1, del_x) + cv2.filter2D(G, -1, del_y)
		second_derivative = cv2.filter2D(first_derivative, -1, del_x) + cv2.filter2D(first_derivative, -1, del_y)

		for orientation in range(orientations):
			filter_orientation = orientation * 180 / orientations
     	
			first_derivative =  imutils.rotate(first_derivative, filter_orientation)
			first_derivatives.append(first_derivative)

			second_derivative = imutils.rotate(second_derivative, filter_orientation)
			second_derivatives.append(second_derivative)
	
	
	for scale in log_scale:
		sigma = sigma = [scale, scale]
		G = gaussian2d(sigma, filter_size)
		log_kernal = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
		LoG.append(cv2.filter2D(G, -1, log_kernal))
		#LoG.append(convolve2d(G, log_kernal))



	for scale in gaussian_scale:
		sigma = [scale, scale]
		gaussian.append(gaussian2d(sigma, filter_size))


	filter_bank = first_derivatives + second_derivatives + LoG + gaussian
	return filter_bank

def gaborFilter(scales, orientations, frequencies, filter_size):
	filter_bank = []
	for scale in scales:
		sigma = [scale, scale]
		G = gaussian2d(sigma, filter_size)
		for frequency in frequencies:
			for orientation in range(orientations):
				filter_orientation = orientation * np.pi / orientations
				sine_wave = sin2d(frequency, filter_size, filter_orientation)
				gabor_filter = G * sine_wave
				filter_bank.append(gabor_filter)

	return filter_bank

def halfdiskFilters(radii, orientations):
	filter_bank = []
	for radius in radii:
		filter_bank_pairs = []
		temp = []
		for orientation in range(orientations):
			angle = orientation * 360 / orientations
			half_disk_filter = halfDisk(radius, angle)
			temp.append(half_disk_filter)

        #to make pairs
		i = 0
		while i < orientations/2:
			filter_bank_pairs.append(temp[i])
			filter_bank_pairs.append(temp[i+int((orientations)/2)])
			i = i+1

		filter_bank+=filter_bank_pairs
	
	
	return filter_bank

def applyFilters(image, filter_bank):
	out_images = []	
	for ft in filter_bank:
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		out_image = cv2.filter2D(image_gray,-1, ft)
		out_images.append(out_image)

	return out_images

def chisquareDistance(input, bins, filter_bank):

	chi_square_distances = []
	N = len(filter_bank)
	n = 0
	while n < N:
		left_mask = filter_bank[n]
		right_mask = filter_bank[n+1]		
		tmp = np.zeros(input.shape)
		chi_sq_dist = np.zeros(input.shape)
		min_bin = np.min(input)
	

		for bin in range(bins):
			tmp[input == bin+min_bin] = 1
			g_i = cv2.filter2D(tmp,-1,left_mask)
			h_i = cv2.filter2D(tmp,-1,right_mask)
			chi_sq_dist += (g_i - h_i)**2/(g_i + h_i + np.exp(-7))

		chi_sq_dist /= 2
		chi_square_distances.append(chi_sq_dist)
		n = n+2
    	

	return chi_square_distances

def getEdges(T_g, B_g, C_g, Canny_edge, Sobel_edges, weights):
	Canny_edge = cv2.cvtColor(Canny_edge, cv2.COLOR_BGR2GRAY)
	Sobel_edges = cv2.cvtColor(Sobel_edges, cv2.COLOR_BGR2GRAY)
	T1 = (T_g + B_g + C_g)/3
	w1 = weights[0]
	w2 = weights[1]
	T2 = (w1 * Canny_edge) + (w2 * Sobel_edges)

	pb_lite_op = np.multiply(T1, T2)
	return pb_lite_op


def main():
	get_textron_maps = True
	get_brightness_map = True
	get_color_map = True

	texture_bins = 64
	brightness_bins = 16
	color_bins = 16
 
	textron_maps = []
	brightness_maps = []
	color_maps = []

	textron_gradients = []
	brightness_gradients = []
	color_gradients = []

	folder_name = "./"
	image_folder_name = folder_name + "data/BSDS500/Images"
	sobel_baseline_folder = folder_name + "data/BSDS500/SobelBaseline"
	canny_baseline_folder = folder_name + "data/BSDS500/CannyBaseline"
	


	print("Generating filters...")
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	dog_filter_bank = DoGFilters(16, [2,3], 49)
	printFilterbank(dog_filter_bank, folder_name + "results/Filters/DoG.png", cols = 8)
	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	LMS_filter_bank = LMFilter([1, np.sqrt(2), 2, 2*np.sqrt(2)], 6, 49)
	printFilterbank(LMS_filter_bank,folder_name + "results/Filters/LMS.png", 6)
	LML_filter_bank = LMFilter([np.sqrt(2), 2, 2*np.sqrt(2), 4], 6, 49)
	printFilterbank(LML_filter_bank, folder_name + "results/Filters/LML.png", 6)
	LM_filter_bank = LMS_filter_bank + LML_filter_bank
	printFilterbank(LM_filter_bank, folder_name + "results/Filters/LM.png", 6)
	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	gabor_filter_bank = gaborFilter([10,25], 6, [2,3,4], 49)
	printFilterbank(gabor_filter_bank, folder_name + "results/Filters/Gabor.png",6)

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	half_disk_filter_bank = halfdiskFilters([2,5,10,20,30], 16)
	printFilterbank(half_disk_filter_bank, folder_name + "results/Filters/HDMasks.png", 6)
	print("generating texton maps..")
	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	edit: used Dog + LM + gabor
	"""	
	images = loadImages(image_folder_name, files=None)
	file_names = os.listdir(image_folder_name)
	filter_bank = dog_filter_bank + LM_filter_bank + gabor_filter_bank
    
	
	if get_textron_maps:
		for i,image in enumerate(images):
			filtered_image = applyFilters(image, filter_bank)	
			filtered_image = np.array(filtered_image)
			f,x,y = filtered_image.shape
			input_mat = filtered_image.reshape([f, x*y])
			input_mat = input_mat.transpose()

			#print(input_mat.shape)
			"""
			Generate texture ID's using K-means clustering
			Display texton map and save image as TextonMap_ImageName.png,
			use command "cv2.imwrite('...)"
			"""
			kmeans = sklearn.cluster.KMeans(n_clusters = texture_bins, n_init = 2)
			kmeans.fit(input_mat)
			labels = kmeans.predict(input_mat)
			texton_image = labels.reshape([x,y])
			textron_maps.append(texton_image)
			#plt.imshow(texton_image)
			#plt.show()			
			plt.imsave(folder_name + "results/Textron_map/TextonMap_"+ file_names[i], texton_image)     

	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	print("generating texton gradients..")
	for i,textron_map in enumerate(textron_maps):
		T_g = chisquareDistance(textron_map, texture_bins, half_disk_filter_bank)
		T_g = np.array(T_g)
		T_g = np.mean(T_g, axis = 0)
		#plt.imshow(T_g)
		#plt.show()	
		textron_gradients.append(T_g)
		plt.imsave(folder_name + "results/T_g/tg_" + file_names[i], T_g)		
		 

    
	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	print("generating brightness maps..")
	if get_brightness_map:
		for i,image in enumerate(images):
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			x,y = image_gray.shape
			input_mat = image_gray.reshape([x*y,1])
			kmeans = sklearn.cluster.KMeans(n_clusters = brightness_bins, n_init = 4)
			kmeans.fit(input_mat)
			labels = kmeans.predict(input_mat)
			brightness_image = labels.reshape([x,y])
			brightness_maps.append(brightness_image)
			#plt.imshow(brightness_image)
			#plt.show()
			plt.imsave(folder_name + "results/Brightness_map/BrightnessMap_" + file_names[i], brightness_image)
   			

	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	print("generating brightness gradient..")
	for i,brightness_map in enumerate(brightness_maps):
		B_g = chisquareDistance(brightness_map, brightness_bins, half_disk_filter_bank)
		B_g = np.array(B_g)
		B_g = np.mean(B_g, axis = 0)
		#plt.imshow(B_g)
		#plt.show()
		brightness_gradients.append(B_g)
		plt.imsave(folder_name + "results/B_g/bg_" + file_names[i], B_g)


	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	print("generating color maps..")
	if get_color_map:
		for i,image in enumerate(images):
			x,y,c = image.shape
			input_mat = image.reshape([x*y,c])
		
			kmeans = sklearn.cluster.KMeans(n_clusters = color_bins, n_init = 4)
			kmeans.fit(input_mat)
			labels = kmeans.predict(input_mat)
			color_image = labels.reshape([x,y])
			color_maps.append(color_image)
			#plt.imshow(color_image)
			#plt.show()			
			plt.imsave(folder_name + "results/Color_map/ColorMap_"+ file_names[i], color_image) 


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	print("generating color gradient..")
	for i,color_map in enumerate(color_maps):
		C_g = chisquareDistance(color_map, color_bins, half_disk_filter_bank)
		C_g = np.array(C_g)
		C_g = np.mean(C_g, axis = 0)
		#plt.imshow(C_g)
		#plt.show()
		color_gradients.append(C_g)
		plt.imsave(folder_name + "results/C_g/cg_" + file_names[i], C_g)


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
	baseline_files = jpg2pngList(file_names)
	print(baseline_files)
	sobel_baseline = loadImages(sobel_baseline_folder, baseline_files)

	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""

	canny_baseline = loadImages(canny_baseline_folder, baseline_files)


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	print("generating pb lite output..")
	if get_textron_maps and get_brightness_map and get_color_map:
		for i in range(len(images)):	
			print("generating edges for image ", baseline_files[i])	
			pb_edge = getEdges(textron_gradients[i], brightness_gradients[i], color_gradients[i], canny_baseline[i], sobel_baseline[i], [0.5,0.5])
			plt.imshow(pb_edge, cmap = "gray")
			plt.show()
			plt.imsave("Phase1/results/pb_lite_output/" + baseline_files[i], pb_edge, cmap = "gray")

if __name__ == '__main__':
    main()
 


