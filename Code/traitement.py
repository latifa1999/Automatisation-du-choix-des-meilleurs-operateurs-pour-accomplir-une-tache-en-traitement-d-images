from flask import Flask , redirect, url_for, render_template, request
from trait import *
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim
from skimage.filters import  roberts
import time
import pickle

class  envi :
	envi 		= cv2.imread("static/men.jpg")
	envi_rgb    = ToRGB(envi)
	img_ref 	= cv2.imread('static/men_seg.jpg')
	img_ref_gr	= ToGRAY(img_ref)

	list_filres  = [median, gaussien, moyenneur ]
	list_segment = [canny, sobel, perwit, laplacien, findContours, kmeans]
	list_morph   = [dilatation, erosion, ouverture, TopHat, cloture, gradient, BlackHat]

	lists = {
		median: "median",
		gaussien: "gaussien",
		moyenneur: "moyenneur",
		canny: "canny",
		sobel: "sobel",
		perwit: "perwit",
		laplacien: "laplacien",
		findContours: "findContours",
		kmeans: "kmeans",
		dilatation: "dilatation",
		erosion: "erosion",
		ouverture: "ouverture",
		TopHat: "TopHat",
		cloture: "cloture",
		gradient: "gradient",
		BlackHat: "BlackHat",
	}

	def action(self):
		filtre  = random.choice(self.list_filres)
		segment = random.choice(self.list_segment)
		morph   = random.choice(self.list_morph)
		actions = [filtre, segment, morph]
		return actions 

	def com(self):
		act  = self.action()
		img  = act[0](self.envi)
		img1 = act[1](img)
		img2 = act[2](img1)
		return img2 ,act

	def ssim(self):
		img_res , action = self.com()
		(score, _) 		 = ssim(img_res, self.img_ref_gr, full=True)
		return self.envi, img_res, action, (format(score))
	
	def reward(self,ssim):
		if float(ssim) >= 0.77:
			return 1
		else:
			return 0 
	
	def liste_act(self):
		actions_list = []
		rewards = []
		s=[]
		for j in range(850):
			img_ini, img_res, actions, ssims = self.ssim()
			if actions not in actions_list:
				actions_list.append(actions)
				nombre_act = actions_list.index(actions)
				s.append(ssims)
				#print(ssims)
				rewards.append(self.reward(ssims))
		return nombre_act,rewards, actions_list, s


app = Flask(__name__)
names=[]

@app.route('/analyse.html')
def analyse():
    return render_template('analyse.html')

@app.route('/')
def home():
	return render_template("index.html")
@app.route('/form', methods=['POST'])
def image():
    if(request.method == "POST"):
        file = request.files['image']
        name = file.filename
        names.append(name)
        if(request.method == "but"):
            img = Image.open("static/"+name)
        return render_template("analyse.html", image=name , image_ref="men_seg.jpg")
@app.route('/form-2', methods=['POST', 'GET'])
def image_res():
	if(request.method == "POST"):
		print("training...")
		Env = envi()
		actions_n,rewards, liste_actions, ss = Env.liste_act()
		status_n = 1
		Q = np.zeros([status_n,actions_n+1])
		lr = 0.1
		y = 0.9
		num_episodes = 100
		s=0
		new = "new.jpg"
		for i in range(num_episodes):
			for k in range(actions_n+1):
				if k==actions_n:
					Q[s, k] = Q[s, k] + lr*(rewards[k] - Q[s, k])
				else:
					Q[s, k] = Q[s, k] + lr*(rewards[k] + y * np.max(Q[s,k+1]) - Q[s, k]) # Fonction de mise à jour de la Q-table
				
		action_opt = np.argmax(Q)
		act  = liste_actions[action_opt]
		combainisnon = (Env.lists[act[0]],Env.lists[act[1]],Env.lists[act[2]])
		img  = act[0](Env.envi)
		img1 = act[1](img)
		img2 = act[2](img1)
		sim = ss[action_opt]
		
		cv2.imwrite("static/"+new,img2)
		return render_template("analyse.html",image=names[0], image_res=new, image_ref="men_seg.jpg", action=combainisnon, ssim = sim)


if __name__ == "__main__":
    app.run(debug = True)
