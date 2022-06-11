# %%
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import cv2
import shutil
import matplotlib
from importlib import reload
# %% The code to calculate line intersections were taken from other authors

# A Python3 program to find if 2 given line segments intersect or not

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def onSegment(p, q, r):
	if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
		(q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
		return True
	return False

def orientation(p, q, r):
	# to find the orientation of an ordered triplet (p,q,r)
	# function returns the following values:
	# 0 : Collinear points
	# 1 : Clockwise points
	# 2 : Counterclockwise
	
	# See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
	# for details of below formula.
	
	val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
	if (val > 0):
		
		# Clockwise orientation
		return 1
	elif (val < 0):
		
		# Counterclockwise orientation
		return 2
	else:
		
		# Collinear orientation
		return 0

# The main function that returns true if
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
	
	# Find the 4 orientations required for
	# the general and special cases
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)

	# General case
	if ((o1 != o2) and (o3 != o4)):
		return True

	# Special Cases

	# p1 , q1 and p2 are collinear and p2 lies on segment p1q1
	if ((o1 == 0) and onSegment(p1, p2, q1)):
		return True

	# p1 , q1 and q2 are collinear and q2 lies on segment p1q1
	if ((o2 == 0) and onSegment(p1, q2, q1)):
		return True

	# p2 , q2 and p1 are collinear and p1 lies on segment p2q2
	if ((o3 == 0) and onSegment(p2, p1, q2)):
		return True

	# p2 , q2 and q1 are collinear and q1 lies on segment p2q2
	if ((o4 == 0) and onSegment(p2, q1, q2)):
		return True

	# If none of the cases
	return False
	
# This code is contributed by Ansh Riyal


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

# This bit is from Stack Overflow

# %%
class neural_network:
    # There are other ways to represent a neural network in a matricial form, but I like this one
    def __init__(self,n1=6,n2=4):
        self.layer1 = 2*np.random.rand(5,n1)-1
        self.bias1  = 2*np.random.rand(1,n1)-1
        self.layer2 = 2*np.random.rand(n1,n2)-1
        self.bias2  = 2*np.random.rand(1,n2)-1
        self.layer3 = 2*np.random.rand(n2,2)-1
        self.bias3  = 2*np.random.rand(1,2)-1
    
    def forward(self,input):
        # The absence of an activation function is equivalent to a linear one
        res = np.matmul(input,self.layer1) + self.bias1
        res = np.matmul(res,self.layer2) + self.bias2
        res = np.matmul(res,self.layer3) + self.bias3
        # Sigmoid for the outputs
        return 1/(1+np.exp(-res[0][0])),10*(1/(1+np.exp(-res[0][1])))-5

# the vehicle is a square of 1*1
class vehicle:
    def __init__(self,init_angle=0):
        self.x = 0  #center of mass
        self.y = 0  #center of mass
        # Vertices of the car
        self.topleft = (self.x + np.sqrt(2)/2*np.cos((init_angle+45)*np.pi/180),self.y + np.sqrt(2)/2*np.sin((init_angle+45)*np.pi/180))
        self.topright = (self.x + np.sqrt(2)/2*np.cos(((init_angle+45)-90)*np.pi/180),self.y + np.sqrt(2)/2*np.sin(((init_angle+45)-90)*np.pi/180))
        self.botleft = (self.x + np.sqrt(2)/2*np.cos(((init_angle+45)+90)*np.pi/180),self.y + np.sqrt(2)/2*np.sin(((init_angle+45)+90)*np.pi/180))
        self.botright = (self.x + np.sqrt(2)/2*np.cos(((init_angle+45)-180)*np.pi/180),self.y + np.sqrt(2)/2*np.sin(((init_angle+45)-180)*np.pi/180))
        self.angle = init_angle
        self.speed = 1
        self.offset = [0.5,np.sqrt(2)/2,0.5,np.sqrt(2)/2,0.5] # distance from the center of mass to the walls of the car that stand in front of the sensor

    def get_distances(self,track):
        # Calculates the distance from the sensors to the surrounding walls 
        sensors = []
        intersect = []
        angles = [90, 45, 0, -45, -90]
        angles = [(self.angle + a)*np.pi/180 for a in angles]
        end_points = [self.x + 100*np.cos(angles),self.y + 100*np.sin(angles)]
        end_points = list(np.array(end_points).transpose())
        for p in end_points:
            dist = 10000
            point = (self.x,self.y)
            for lines in track:
                if doIntersect(Point(self.x,self.y),Point(p[0],p[1]),Point(lines[0],lines[1]),Point(lines[2],lines[3])):
                    x,y = line_intersection([(self.x,self.y),(p[0],p[1])],[(lines[0],lines[1]),(lines[2],lines[3])])
                    if np.sqrt((x-self.x)*(x-self.x)+(y-self.y)*(y-self.y))<dist:
                        dist = np.sqrt((x-self.x)*(x-self.x)+(y-self.y)*(y-self.y))
                        point = (x,y)
            sensors.append(dist)
            intersect.append(point)
        for i in range(5):
            sensors[i] -= self.offset[i]
        sensors = [min(15,s)/7.5 -1 for s in sensors]

        return sensors,intersect
    
    def check_collisions(self,track):
        # Checks if the track limits intersect the car (assumption that the car cant move fast enough to jump the limits)
        for lines in track:
                if doIntersect(Point(self.topleft[0],self.topleft[1]),Point(self.topright[0],self.topright[1]),Point(lines[0],lines[1]),Point(lines[2],lines[3])):
                    return True
                elif doIntersect(Point(self.topleft[0],self.topleft[1]),Point(self.botleft[0],self.botleft[1]),Point(lines[0],lines[1]),Point(lines[2],lines[3])):
                    return True
                elif doIntersect(Point(self.botleft[0],self.botleft[1]),Point(self.botright[0],self.botright[1]),Point(lines[0],lines[1]),Point(lines[2],lines[3])):
                    return True
                elif doIntersect(Point(self.topright[0],self.topright[1]),Point(self.botright[0],self.botright[1]),Point(lines[0],lines[1]),Point(lines[2],lines[3])):
                    return True
        return False
    
    def update_position(self,delta,matrix,track):
        # New location of the vehicle after one time step 
        input,_ = self.get_distances(track)
        speed,delta_ang = matrix.forward(input)
        self.angle += delta_ang
        self.x += speed*delta*np.cos(self.angle*np.pi/180)
        self.y += speed*delta*np.sin(self.angle*np.pi/180)
        ref_angle = self.angle + 45
        self.topleft = (self.x + np.sqrt(2)/2*np.cos(ref_angle*np.pi/180),self.y + np.sqrt(2)/2*np.sin(ref_angle*np.pi/180))
        self.topright = (self.x + np.sqrt(2)/2*np.cos((ref_angle-90)*np.pi/180),self.y + np.sqrt(2)/2*np.sin((ref_angle-90)*np.pi/180))
        self.botleft = (self.x + np.sqrt(2)/2*np.cos((ref_angle+90)*np.pi/180),self.y + np.sqrt(2)/2*np.sin((ref_angle+90)*np.pi/180))
        self.botright = (self.x + np.sqrt(2)/2*np.cos((ref_angle-180)*np.pi/180),self.y + np.sqrt(2)/2*np.sin((ref_angle-180)*np.pi/180))
        

def mix_networks(mat,list):
    # The genetic part: it randomly attributes genes of individuals that perform well to those who don't 
    new_mat = []
    for i in range(len(mat)):
        new_mat.append(mat[list[i]])

    ## Why np.array()? Well, if you say that a part of one numpy matrix is equal to another part of another numpy matrix enough times, shit will eventually happen
    # You could probably also cast it to float(or a variant of it), try it if you feel like it
      
    for i in range(5,len(mat)-5):
        for j in range(mat[0].layer1.shape[0]):
            for k in range(mat[0].layer1.shape[1]):
                if random.random()>0.6:
                    new_mat[i].layer1[j][k] = np.array(new_mat[len(mat)-1-i].layer1[j][k])
                if j==1 and random.random()>0.6:
                    new_mat[i].bias1[j-1][k] = np.array(new_mat[len(mat)-1-i].bias1[j-1][k])
        for j in range(mat[0].layer2.shape[0]):
            for k in range(mat[0].layer2.shape[1]):
                if random.random()>0.6:
                    new_mat[i].layer2[j][k] = np.array(new_mat[len(mat)-1-i].layer2[j][k])
                if j==1 and random.random()>0.6:
                    new_mat[i].bias2[j-1][k] = np.array(new_mat[len(mat)-1-i].bias2[j-1][k])
        for j in range(mat[0].layer3.shape[0]):
            for k in range(mat[0].layer3.shape[1]):
                if random.random()>0.6:
                    new_mat[i].layer3[j][k] = np.array(new_mat[len(mat)-1-i].layer3[j][k])
                if j==1 and random.random()>0.6:
                    new_mat[i].bias3[j-1][k] = np.array(new_mat[len(mat)-1-i].bias3[j-1][k])
    
    for i in range(len(mat)-5,len(mat)):
        for j in range(mat[0].layer1.shape[0]):
            for k in range(mat[0].layer1.shape[1]):
                if random.random()>0.3:
                    new_mat[i].layer1[j][k] = np.array(new_mat[len(mat)-1-i].layer1[j][k])
                if j==1 and random.random()>0.6:
                    new_mat[i].bias1[j-1][k] = np.array(new_mat[len(mat)-1-i].bias1[j-1][k])
        for j in range(mat[0].layer2.shape[0]):
            for k in range(mat[0].layer2.shape[1]):
                if random.random()>0.3:
                    new_mat[i].layer2[j][k] = np.array(new_mat[len(mat)-1-i].layer2[j][k])
                if j==1 and random.random()>0.6:
                    new_mat[i].bias2[j-1][k] = np.array(new_mat[len(mat)-1-i].bias2[j-1][k])
        for j in range(mat[0].layer3.shape[0]):
            for k in range(mat[0].layer3.shape[1]):
                if random.random()>0.3:
                    new_mat[i].layer3[j][k] = np.array(new_mat[len(mat)-1-i].layer3[j][k])
                if j==1 and random.random()>0.6:
                    new_mat[i].bias3[j-1][k] = np.array(new_mat[len(mat)-1-i].bias3[j-1][k])
                
    return new_mat


def checkpoint_status(coordinates, cond):
    # Evaluates the completion of the track
    if cond[0] == 'x':
        if cond[1]<cond[2]:
            if coordinates[0]>=cond[2]:
                return cond[3]
            else:
                return cond[3]*(coordinates[0]-cond[1])/(cond[2]-cond[1])
        else:
            if coordinates[0]<=cond[2]:
                return cond[3]
            else:
                return cond[3]*(coordinates[0]-cond[1])/(cond[2]-cond[1])
    else:
        if cond[1]<cond[2]:
            if coordinates[1]>=cond[2]:
                return cond[3]
            else:
                return cond[3]*(coordinates[1]-cond[1])/(cond[2]-cond[1])
        else:
            if coordinates[1]<=cond[2]:
                return cond[3]
            else:
                return cond[3]*(coordinates[1]-cond[1])/(cond[2]-cond[1])

def init_config():
    ## Configure your own experience. The code is ugly but it makes sure you don't mess up with it.

    if os.path.isdir('gif'):
        shutil.rmtree('gif')
    print('Welcome, Random User!')
    print('There are two tracks you can choose from:')
    print('1. Slalom\n2. Oval')
    opt = input('Pick one of them: ')
    if opt=='':
        opt=0
    else:
        opt = int(opt)-1
        if opt>1 or opt<0:
            opt = 0

    inp = input('Do you wish to personalize the neural network?(y/n)')
    if inp=='y':
        n1 = input('Neurons on Hidden Layer 1: ')
        if n1=='':
            n1=6
        else:
            n1 = int(n1)
            if n1<1 or n1>20:
                n1 = 6
        n2 = input('Neurons on Hidden Layer 2: ')
        if n2=='':
            n2=4
        else:
            n2 = int(n2)
            if n2<1 or n2>20:
                n2 = 4
    else:
        n1 = 6
        n2 = 4
    
    ep = input('Number of epochs for training (at least 20): ')
    if ep=='':
        ep = 20
    else:
        ep = int(ep)
        if ep<20 or ep>10000:
            ep = 20
    
    pop = input('Size of the population (at least 20, multiple of 5): ')
    if pop=='':
        pop = 20
    else:
        pop = int(pop)
        if pop<20 or pop>10000 or (pop%5)!=0:
            pop = 20

    vid = input('Do you want to generate a video of the training process?(y/n) ')
    if vid=='y':
        vid = True
        os.makedirs('gif')
        # Matplotlib gets a brain fart if you plt.save a lot of stuff (not using a notebook smh). This solves it, thanks Stack Overflow
        reload(matplotlib)
        matplotlib.use('Agg')
    else:
        vid = False  

    return opt,n1,n2,ep,pop,vid

# %%
def generate_video():
    images = [img for img in os.listdir('gif') if img.endswith(".png")]
    images = [x.replace('.png', '') for x in images]
    images.sort(key=int)
    images = [x + '.png' for x in images]
    frame = cv2.imread(os.path.join('gif', images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('output.mp4', fourcc, 10, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join('gif', image)))
    video.release()
    shutil.rmtree('gif')

# %%
opt,n1,n2,epoch,population,print_order = init_config()
mat = []
results = []
img = []
lastc = 0
maxx = 0
minn = 1000

# The track is defined by several line segments that represent the walls
track_list = [[(-2,0,-2,-4),(-2,0,8,10),(-2,-4,8,6),(8,10,16,2),(8,6,16,-2),(16,2,24,10),(16,-2,24,6),(24,10,34,0),(24,6,34,-4)],
            [(-7,2,7,2),(-5,-2,5,-2),(7,2,13,-2),(5,-2,9,-4),(9,-4,9,-6),(13,-2,13,-8),(13,-8,7,-12),(9,-6,5,-8),(7,-12,-7,-12),(5,-8,-5,-8),(-7,2,-13,-2),(-5,-2,-9,-4),(-9,-4,-9,-6),(-13,-2,-13,-8),(-13,-8,-7,-12),(-9,-6,-5,-8)]]

# checkpoints indicate starting x or y coordinate, the one we are trying to reach and the percentage of the track completion that it adds once it is reached
checkpoint_list = [[('x',0,8,0.25),('x',8,16,0.25),('x',16,24,0.25),('x',24,32,0.25)],
                [('x',0,6,0.125),('y',0,-3,0.1),('y',-3,-7,0.05),('x',11,6,0.1),('x',6,-6,0.25),('y',-10,-7,0.1),('y',-7,-3,0.05),('x',-11,-6,0.1),('x',-6,0,0.125)]]

# Initial angle of the vehicle
init_angle = [45,0]

track = track_list[opt]
checkpoints = checkpoint_list[opt]
ang = init_angle[opt]

for i in range(population):
    mat.append(neural_network(n1,n2))

for e in range(epoch):
    for m in mat:
        v = vehicle(ang)
        c = 0
        check = 0
        check_id = 0
        while(not(v.check_collisions(track)) and check_id<len(checkpoints) and c<500):
            v.update_position(0.5,m,track)
            temp = checkpoint_status((v.x,v.y),checkpoints[check_id])
            if temp == checkpoints[check_id][3]:
                check+=checkpoints[check_id][3]
                check_id+=1
            c+=1

        if check<1:
            check+=temp
        results.append([check,c])
    l2 = [results.index(x) for x in sorted(results,key= lambda l: (-l[0],l[1]))]
    print('Epoch '+ str(e+1)+': ', results[l2[0]])
    
    if ((maxx<results[l2[0]][0] and maxx<1) or (results[l2[0]][0]==1 and results[l2[0]][1]<minn)) and print_order:
        maxx = results[l2[0]][0]
        if maxx == 1:
            minn = results[l2[0]][1]
        m = mat[l2[0]]
        c=0
        v = vehicle(ang)

        while(not(v.check_collisions(track)) and c<results[l2[0]][1]):
            _,intersection = v.get_distances(track)
            fig = plt.figure(facecolor='white')
            for t in track:
                plt.plot((t[0],t[2]),(t[1],t[3]),color='red')
            plt.plot((v.topleft[0],v.topright[0]),(v.topleft[1],v.topright[1]),color='blue')
            plt.plot((v.botleft[0],v.botright[0]),(v.botleft[1],v.botright[1]),color='blue')
            plt.plot((v.topleft[0],v.botleft[0]),(v.topleft[1],v.botleft[1]),color='blue')
            plt.plot((v.botright[0],v.topright[0]),(v.botright[1],v.topright[1]),color='blue')
            plt.title('Epoch ' + str(e+1))
            plt.axis('off')
            for i in intersection:
                if i[0]!=v.x and i[1]!=v.y:
                    plt.plot((v.x,i[0]),(v.y,i[1]),color='orange')
            fig.savefig('./gif/'+str(lastc+c)+'.png',facecolor=fig.get_facecolor(),transparent=False)
            plt.close('all')
            v.update_position(0.5,m,track)
            c+=1
        lastc += c
    
    mat = mix_networks(mat.copy(),l2).copy()
    results = []

if print_order:
    generate_video()
