import numpy as np 

class operation:
	def convolution(x,no_of_filters,filter_size,stride=1,pad=0):
		#padding
		new_image_size=(x.shape[1])+(2*pad)
		y=np.zeros(shape=(3,new_image_size,new_image_size))
		for i in range(3):
			y[i,pad:new_image_size-pad,pad:new_image_size-pad]=x[i]
		#filter
		f=np.random.uniform(size=(no_of_filters,3,filter_size,filter_size))		
		#output
		output_size=int(((new_image_size-filter_size)/stride)+1)
		output=np.zeros(shape=(no_of_filters,output_size,output_size))
		for k in range(0,no_of_filters,1):
			row=0
			for i in range(0,output_size*stride,stride):
				col=0
				for j in range(0,output_size*stride,stride):
					iter=y[:,i:filter_size+i,j:filter_size+j]
					convolve=np.sum(iter*f[k,:,:,:])
					output[k,row,col]=convolve
					col+=1
				row+=1
		return output		

	def max_pooling(x,pooling_size=2,stride=2):	
		#output
		output_size=int(((x.shape[1]-pooling_size)/stride)+1)
		output=np.zeros(shape=(x.shape[0],output_size,output_size))
		for k in range(0,x.shape[0],1):
			row=0
			for i in range(0,output_size*stride,stride):
				col=0
				for j in range(0,output_size*stride,stride):
					output[k,row,col]=np.max(x[k,i:pooling_size+i,j:pooling_size+j])
					col+=1
				row+=1
		return output			

	def activation(x):
		for k in range(x.shape[0]):
			for i in range(x.shape[1]):
				for j in range(x.shape[2]):
					if(x[k,i,j]<0):
						x[k,i,j]=0
		return x	
	
	def flattening(x):
		output_size=x.shape[0]*x.shape[1]*x.shape[2]
		output=np.zeros(shape=(output_size,1))
		h=0
		for k in range(x.shape[0]):
			for i in range(x.shape[1]):
				for j in range(x.shape[2]):
					output[h]=x[k,i,j]
					h+=1
		return output

	def tanh(x):
		return np.tanh(x)				

	def forward_propagation(x,hiddenlayer=6,output=1):
		w0=np.random.uniform(size=(x.shape[0],hiddenlayer))#weight of image and hiddenlayer
		w1=np.random.uniform(size=(hiddenlayer,output))#weight of hiddenlayer and output
		b0=np.random.uniform(size=(hiddenlayer,1))#bias applied in hiddenlayer
		b1=np.random.uniform(size=(output,1))#bias applied in output	
		h1=np.dot(w0.T,x)+b0#hiddenlayer input
		h2=operation.tanh(h1)#hiddenlayer output
		O1=np.dot(w1.T,h2)+b1#outputlayer input
		O2=operation.tanh(O1)#output
		return O2

	def softmax(x):
		return np.exp(x)/(np.sum(np.exp(x)))





