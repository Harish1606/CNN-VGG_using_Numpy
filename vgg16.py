from layer import operation

class Vgg:
	def function(x):
		conv_output1=operation.convolution(x,no_of_filters=64,filter_size=3,stride=1,pad=1)
		pooling_layer1=operation.max_pooling(conv_output1,pooling_size=2,stride=2)
		
		conv_output2=operation.convolution(pooling_layer1,no_of_filters=128,filter_size=3,stride=1,pad=1)
		pooling_layer2=operation.max_pooling(conv_output2,pooling_size=2,stride=2)

		conv_output3=operation.convolution(pooling_layer2,no_of_filters=256,filter_size=3,stride=1,pad=1)
		pooling_layer3=operation.max_pooling(conv_output3,pooling_size=2,stride=2)

		conv_output4=operation.convolution(pooling_layer3,no_of_filters=512,filter_size=3,stride=1,pad=1)
		pooling_layer4=operation.max_pooling(conv_output4,pooling_size=2,stride=2)

		conv_output5=operation.convolution(pooling_layer4,no_of_filters=512,filter_size=3,stride=1,pad=1)
		pooling_layer5=operation.max_pooling(conv_output5,pooling_size=2,stride=2)

		flattening_output1=operation.flattening(pooling_layer5)
		print(flattening_output1.shape)

		fully_connected1=operation.forward_propagation(flattening_output1,hiddenlayer=25088,output=25088)
		fully_connected2=operation.forward_propagation(fully_connected1,hiddenlayer=25088,output=4096)
		fully_connected3=operation.forward_propagation(fully_connected2,hiddenlayer=2000,output=1000)
		
		Softmax=operation.softmax(fully_connected1)
		#return(Softmax)



		

