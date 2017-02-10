import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fname = "velib.npz"
## fonction de deserialization de numpy
obj = np.load(fname)
## objets contenus dans le fichier
print(obj.keys())
## matrice 1217x#minutes nombre de velos disponibles
histo = obj['histo']
## matrice 1217x#minutes, pour chaque station nombre de velib pris a chaque minute
take = obj['take']
## infos stations statiques:
#### id_velib->(nom,addresse,coord_y,coord_x,banking,bonus,nombre de places
stations = dict(obj['stations'].tolist())
## id_velib -> id matrice take, histo
idx_stations = dict(obj['idx_stations'].tolist())
stations_idx = dict(obj['stations_idx'].tolist()) ## id matrice -> velib

plt.ion()
parismap = mpimg.imread('paris-48.806-2.23--48.916-2.48.jpg')
## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan

geo_mat = np.zeros((len(idx_stations),2))
for i,idx in idx_stations.items():
    geo_mat[i,0] =stations[idx][3]
    geo_mat[i,1]= stations[idx][2]
## alpha permet de regler la transparence
plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.3)


# discretisation
steps = 100
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]

# A remplacer par res = monModele.predict(grid).reshape(steps,steps)
res = np.random.random((steps,steps))
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,0],geo_mat[:,1],alpha=0.3)
