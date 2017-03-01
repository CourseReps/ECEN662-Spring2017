from PIL import Image
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.stats as stats
#print(os.getcwd())


def image_values(x):
    im_test = Image.open(x)
    basewidth = 300
    hsize = 150
    i = 0
    im_n = im_test.resize((basewidth, hsize), Image.ANTIALIAS)
    im_n = np.array(im_n.histogram())
    im_n = im_n 
    im_test_pdf = list(im_n[0:768])
    max_value = max(im_test_pdf)
    max_index = im_test_pdf.index(max_value)
    return(max_value)
    
def real_test(x):
    x=image_values(x)
    print(x)
    x=np.linspace(0,10000,500)
    h1=stats.norm.pdf(x,np.mean(real_values),np.var(real_values))
    h0=stats.norm.pdf(x,np.mean(fake_values),np.var(fake_values))
    if h1/h0>1:
        return 1
    else:
        return 0
        
def ktest(x):
    return 0
    

real = os.listdir(r"C:\Users\Digveer\Desktop\662\TrainingSetScenes")
fake = os.listdir(r"C:\Users\Digveer\Desktop\662\TrainingSetSynthetic")
realpath=r"C:\Users\Digveer\Desktop\662\TrainingSetScenes"+'\\'
fakepath=r"C:\Users\Digveer\Desktop\662\TrainingSetSynthetic"+'\\'
real=[realpath+ s for s in real]
#print(real)
fake=[fakepath+ s for s in fake]
#print(fake)

real_values=[]
fake_values=[]
for x in real:
    real_values.append(image_values(x))

for x in fake:
    fake_values.append(image_values(x))

#There has to be a better way to incrememnt stuff. 
real_tot=0
fake_tot=0

for x in real:
    real_tot=real_tot+real_test(x)
    
for x in fake:
    fake_tot=fake_tot+real_test(x)
    
print('Real images detected correctly: ', real_tot/len(real))
print('Fake images detected correctly: ', 1-(fake_tot/len(fake)))

plt.plot(real_values) 
plt.hold()
plt.plot(fake_values)
    