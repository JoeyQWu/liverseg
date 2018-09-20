deep structured active contour model to do liver segmentation

network architecture:
layers = 6,
numfilt= [32,64,128,128,256,256]
MLP to predict the energy,the alpha,the beta,the kappa:[64,256]
input size : [512,512]
output size : [256,256]

Energy = Data term + Eint + Balloon term

manually initialization 
