import sys
import os
import numpy as np 
import tensorflow as tf 
from keras.applications.xception import Xception
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from PIL import Image 
import argparse

def parse_args():
	args = argparse.ArgumentParser()
	args.add_argument("--rootdir", 					type=str, default='/data/')
	args.add_argument("--train_data_dir", 			type=str, default="train")
	args.add_argument("--validation_data_dir", 		type=str, default="validation")
	args.add_argument("--test_data_dir",			type=str, default='test')
	args.add_argument("--test_list",				type=str, default="test.csv")
	args.add_argument("--model_path", 				type=str, default="model")
	args.add_argument("--pretrain",					type=str, default="true")
	args.add_argument("--weights",					type=str, default='xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
	args.add_argument("--mode",						type=str, default="train")
	args.add_argument("--nb_classes", 				type=int, default=2)
	args.add_argument("--batch_size", 				type=int, default=8)
	args.add_argument("--nb_epochs", 				type=int, default=50)
	args.add_argument("--height", 					type=float, default=300)
	args.add_argument("--width", 					type=float, default=200)
	args.add_argument("--learning_rate", 			type=float, default=1e-4)
	args.add_argument("--momentum", 				type=float, default=0.9)
	args.add_argument("--transformation_ratio", 	type=float, default=0.2)
	args = vars(args.parse_args())
	print('[INFO] Parameters: ')
	print("="*40)
	for k in sorted(args.keys()):
		print("%20s : %-20s" % (k, args[k]))
	print("="*40)
	return args

def train(args):
	h 					 = args['height']
	w 				     = args['width']
	nb_classes 			 = args['nb_classes']
	transformation_ratio = args['transformation_ratio']
	train_data_dir 		 = args['rootdir'] + args['train_data_dir']
	validation_data_dir  = args['rootdir'] + args['validation_data_dir']
	model_path 			 = args['rootdir'] + args['model_path']
	batch_size 			 = args['batch_size']
	nb_epochs 			 = args['nb_epochs']
	momentum 			 = args['momentum']
	lr 				     = args['learning_rate']
	weights 			 = args["rootdir"] + args['weights']

	classes = {}
	print("[INFO] Classes indices: ")
	print("="*40)
	for idx, f in enumerate(sorted(os.listdir(train_data_dir))):
		classes[f] = idx
		print("%20s : %-20s" % (f, idx))
	print("="*40)

	# create the base pre-trained model
	if args['pretrain'] == 'true':
		base_model = Xception(input_shape=(h, w, 3), weights=weights, include_top=False)
	else:
		base_model = Xception(input_shape=(h, w, 3), weights=None, include_top=False)
	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	predictions = Dense(nb_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	# compile the model (should be done *after* setting layers to non-trainable)
	train_datagen = ImageDataGenerator(rescale=1/255.,
							rotation_range=transformation_ratio*100.0,
							shear_range=transformation_ratio,
							zoom_range=transformation_ratio,
							horizontal_flip=True,
							vertical_flip=True)
	validation_datagen = ImageDataGenerator(rescale=1/255.)

	if not os.path.exists(model_path):
		os.mkdir(model_path)
	train_generator = train_datagen.flow_from_directory(train_data_dir,
							target_size=(h, w),
							batch_size=batch_size,
							class_mode='categorical',
							seed=42,
							classes=classes)
	validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
							target_size=(h, w),
							batch_size=batch_size,
							class_mode='categorical',
							seed=42,
							classes=classes)
	model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['acc'])

	top_weights_path = os.path.join(model_path, 'model_weights.h5')
	callbacks_list = [
			ModelCheckpoint(top_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
			EarlyStopping(monitor='val_loss', patience=15, verbose=1),
			ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=5, verbose=1)
			]

	print("[INFO] Starting to train model...")
	# train the model on the new data for a few epochs
	model.fit_generator(train_generator,
						steps_per_epoch=train_generator.shape[0]//batch_size,
						epochs=nb_epochs,
						validation_data=validation_generator,
						validataion_steps=validation_generator.shape[0],
						callbacks=callbacks_list)

	# save model
	model_json = model.to_json()
	with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
		json_file.write(model_json)

def test(args):
	# load model
	model_path = args["rootdir"] + args['model_path']
	model = load_model(os.path.join(model_path, "model_weights.h5"))
	print("[INFO] Model loading success. Start testing... ")

	# prediction
	h = args['height']
	w = args['width']
	test_list = args['rootdir'] + args['test_list']
	corr = 0
	wron = 0
	with open(test_list, 'r') as f:
		for l in f.readlines():
			img, cl = l.split(",")
			imgname = img
			cl = int(cl)
			img = Image.open(img).convert('RGB')
			img = img.resize((w,h), Image.ANTIALIAS)
			img = np.array(img) / 255.
			img = np.expand_dims(img, 0)
			pred = (model.predict(img))[0].argmax()
			if pred == cl:
				corr += 1
				outp = "correct"
			else:
				wron += 1
				outp = "wrong"
			print("[INFO] Image {} prediction: {}. Label: {}, prediction: {}".format(imgname, 
																					 outp,
																					 cl,
																					 pred))
	print("[INFO] Test accuracy: {}".format(float(corr)/(corr+wron)))

if __name__ == "__main__":
	args = parse_args()

	if args["mode"] == "train":
		train(args)
	elif args["mode"] == "test":
		test(args)
	else:
		ValueError("Mode {} is not applicable. ".format(args["mode"]))

	K.clear_session()