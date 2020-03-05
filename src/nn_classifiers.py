import tensorflow as tf
from tf.keras import layers
from tf.keras.regularizers import l2 as l2_reg
from tf.keras.callbacks import EarlyStopping

class MultiLabelMLP:

	def __init__(self,layers=None,regularizer=l2_reg(1e-05),loss=None,
				max_epochs=20,batch_size=32,val_split=0.1,
				early_stopping_patience=3,debug=False,):
		if layers is None:
			self.layers=[{"size": 1000, "activation" : "relu"}]
		else:
			self.layers=layers
		self.regularizer=regularizer
		self.loss= "binary_crossentropy" if loss is None else loss
		self.max_epochs=max_epochs
		self.batch_size=batch_size
		self.val_split=val_split
		self.debug=debug
		if val_split >0 and early_stopping_patience>0:
			self.callbacks=[EarlyStopping(monitor='val_loss', patience=early_stopping_patience)]

	def fit(self,x_mat,y_mat):
		assert(x_mat.shape[0]==y_mat.shape[0])
		self.num_features=x_mat.shape[1]
		self.num_labels=y_mat.shape[1]
		# build keras model
		inp_x = tf.keras.Input(shape=(self.num_features),name="inp_x")
		x=inp_x
		for i in range(0,len(self.layers)):
			x=layers.Dense(self.layers[i]["size"],activation=self.layers[i]["activation"],
							kernel_regularizer=self.regularizer,name="dense_"+str(i))(x)
		output=layers.Dense(self.num_labels,activation="sigmoid")(x)
		model=tf.keras.Model(inputs=inp_x, outputs=output)
		if self.debug:
			model.summary()
		self.model=model
		self.model.compile(loss=self.loss, optimizer="adam")
		# fit model
		if self.val_split>0:
			self.model.fit(	x_mat,y_mat,epochs=self.max_epochs, 
							batch_size=self.batch_size,
							validation_split=self.val_split,
							callbacks=self.callbacks,verbose=int(self.debug))
		else:
			self.model.fit(x_mat,y_mat, epochs=self.max_epochs,
							batch_size=self.batch_size,
							verbose=int(self.debug))

	def predict_proba(self,x_tst):
		assert(x_tst.shape[1]==self.num_features)
		return self.model.predict(x_tst)
	
	def predict(self,x_tst,threshold=0.5,return_probs=False):
		probs=self.predict_proba(x_tst)
		labels=(probs>threshold)*1
		if return_probs:
			return labels,probs
		else:
			return labels