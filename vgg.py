#importing packages
import numpy as np

from vgg16 import Vgg

np.random.seed(0)
#input
x=np.random.uniform(size=(3,224,224))

emp=Vgg.function(x)
#print(emp)


