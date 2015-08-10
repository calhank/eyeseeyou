import time
import pickle
import numpy as np
import theano
import theano.tensor as T
import lasagne
from matplotlib import pyplot as plt

import os
from pandas import read_csv
import sklearn

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    FTRAIN = 'dataraw/training.csv'
    FTEST = 'dataraw/test.csv'
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
#     df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = sklearn.utils.shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
    	return X
        y = None
        
    print X.shape
    print y.shape

    return X, y

# train_images, train_labels = load()
# print train_images.shape
# print train_labels.shape

# start = time.time()
# with open('faces.pkl','wb') as pk:
#     pickle.dump((train_labels, train_images), pk)
# print time.time() - start
# start = time.time()
# with open('faces.pkl','rb') as pk:
#     train_labels, train_images = pickle.load(pk)
# print time.time() - start



def xy_to_index(x_coord, y_coord, rows=96, cols=96):
    """ assumes 0 index on both x and y axes """
    x_coord, y_coord = int(x_coord), int(y_coord)
    index = x_coord + cols * y_coord
    if x_coord >= cols or y_coord >= rows:
        raise Exception('Index not within image limits', index)
    return int(index)

def index_to_xy(index, rows=96, cols=96):
    y,x = divmod(index, cols)
    return (x,y)


def load_data(fname = 'faces.pkl'):
	with open(fname) as pk:
		return pickle.load(pk)

def build_mlp(input_var = None):
	l_in = lasagne.layers.InputLayer(shape=(None, 9216), input_var=input_var)
	l_in_drop = lasagne.layers.DropoutLayer(l_in, p=.2)

	l_hid1 = lasagne.layers.DenseLayer(l_in_drop,
		num_units=200,
		# nonlinearity=lasagne.nonlinearities.rectify,
		W=lasagne.init.GlorotUniform())

	l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=.2)

	l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop,
		num_units=200,
		# nonlinearity=lasagne.nonlinearities.rectify
		)

	l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=.2)

	l_hid3 = lasagne.layers.DenseLayer(l_hid2_drop,
		num_units=200,
		# nonlinearity=lasagne.nonlinearities.rectify
		)

	l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=.2)

	l_hid4 = lasagne.layers.DenseLayer(l_hid3_drop,
		num_units=100,
		# nonlinearity=lasagne.nonlinearities.rectify
		)

	l_hid4_drop = lasagne.layers.DropoutLayer(l_hid4, p=.2)

	l_hid5 = lasagne.layers.DenseLayer(l_hid4_drop,
		num_units=100,
		# nonlinearity=lasagne.nonlinearities.rectify
		)

	l_out= lasagne.layers.DenseLayer(
		l_hid5,
		num_units=30,
		# nonlinearity=lasagne.nonlinearities.rectify
		)

	return l_out




def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(train_x, train_y, val_x, val_y, test_x, model='mlp', num_epochs=500):
	input_var = T.matrix('inputs')
	target_var = T.fmatrix('targets')
	print target_var

	print 'Building model and compiling functions'

	if model=='mlp':
		network = build_mlp(input_var=input_var)

	print 1
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.squared_error(prediction, target_var)
	loss = loss.mean()

	print 2
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.001, momentum=0.9)

	print 3
	test_prediction=lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.squared_error(test_prediction, target_var)

	test_loss = test_loss.mean()

	print 4
	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

	def test_accuracy(test_prediction=test_prediction, target_var=target_var):
		print test_prediction
		print target_var
		accuracy = T.sqrt(T.mean((test_prediction - target_var)**2))
		print accuracy
		return accuracy

	# test_acc = test_accuracy()
	# test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var)),
                      # dtype=theano.config.floatX)

	print 5
	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], test_loss)#, allow_input_downcast=True)

	test_fn = theano.function([input_var], test_prediction)
	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
	    # In each epoch, we do a full pass over the training data:
	    train_err = 0
	    train_batches = 0
	    start_time = time.time()
	    for batch in iterate_minibatches(train_x, train_y, 200, shuffle=True):
	        inputs, targets = batch
	        train_err += train_fn(inputs, targets)
	        train_batches += 1

	    # And a full pass over the validation data:
	    val_err = 0
	    # val_acc = 0
	    val_batches = 0
	    for batch in iterate_minibatches(val_x, val_y, 50, shuffle=False):
	        inputs, targets = batch
	        # print inputs
	        # print 
	        # print targets

	        # raise Exception
	        err = val_fn(inputs, targets)
	        # print err, acc
	        val_err += err
	        # val_acc += acc
	        val_batches += 1
	        # print val_batches


	    # Then we print the results for this epoch:
	    print("Epoch {} of {} took {:.3f}s".format(
	        epoch + 1, num_epochs, time.time() - start_time))
	    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
	    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
	    # print("  validation accuracy:\t\t{:.2f} %".format(
	        # val_acc / val_batches * 100))

	
	# After training, we compute and print the test error:
	# test_err = 0
	# test_acc = 0
	# test_batches = 0
	# for batch in iterate_minibatches(test_x, test_y, 50, shuffle=False):
	#     inputs, targets = batch
	#     err = val_fn(inputs, targets)
	#     test_err += err
	#     # test_acc += acc
	#     test_batches += 1
	# print("Final results:")
	# print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	# print("  test accuracy:\t\t{:.2f} %".format(
	    # test_acc / test_batches * 100))
	with open('model2results.pkl','wb') as out:
		data = test_fn(test_x)
		pickle.dump(data, out)

	with open('model2.pkl','wb') as out2:
		params = lasagne.layers.get_all_param_values(network)
		pickle.dump(params, out2)

	print 'Done!'



if __name__=="__main__":
	train_labels, train_data, test_data = load_data()
	# test_data = load(test=True)
	# with open('faces.pkl','wb') as pk:
	#     pickle.dump((train_labels, train_data, test_data), pk)

	print train_data.shape
	print train_labels[0].shape
	print train_labels.shape

	train_labels = np.array([ (i+1)/2 for i in train_labels])
	print train_labels[0].shape
	print train_labels.shape

	# train_data = train_data.reshape(-1,1,96,96)
	# test_data = test_data.reshape(-1,1,96,96)
	# print train_data.shape

	# plt.imshow(train_data[0].reshape(96,96), cmap='gray')

	# crds = [ train_labels[0][k:k+2]*48 + 48 for k in xrange(0,30,2)]
	# for crd in crds:
	# 	plt.plot( crd[0], crd[1], 'co' )
	# plt.show()
	

	main(train_data[:-200], train_labels[:-200], train_data[-200:], train_labels[-200:], test_data, num_epochs=5000)

	with open('model2results.pkl','rb') as pk:
		predictions = pickle.load(pk)

	n=2

	plt.imshow(test_data[n].reshape(96,96), cmap='gray')
	for crd in [ predictions[n][i:i+2]*96 for i in xrange(0,len(predictions[n]),2) ]:
		plt.plot(crd[0], crd[1], 'co')
	plt.show()


