import numpy as np
import scipy
import scipy.io
import StringIO
import cv2
import matplotlib.pyplot as plt
import time


from rover import Rover20



class MPCR_Rover(Rover20):
	def __init__(self):
		Rover20.__init__(self)
		self.currentImage = None
		self.quit = False
		self.action_choice = 1
		self.action_labels =['left','forward','right','backward']
		self.action_vectors =[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]		


	



		

		self.n1=240*320+1
		self.n2=6
		self.n3=4

		self.w1=0.005*(1-np.random.random((self.n1,self.n2-1)))
		self.w2=0.005*(1-np.random.random((self.n2,self.n3)))

		self.dw1=np.zeros(self.w1.shape)
		self.dw2=np.zeros(self.w2.shape)

		self.L=0.01
		self.M=0.5

		self.loop=0
		self.sse=10 













	def af(self,x):
		return [1/(1+np.exp(-x))]







	def mleft(self):
		self.setTreads(-1,1)
		time.sleep(.1)
		self.setTreads(0,0)



	def mforward(self):
		self.setTreads(1,1)
		time.sleep(.1)
		self.setTreads(0,0)
		


	def mright(self):
		self.setTreads(1,-1)
		time.sleep(.1)
		self.setTreads(0,0)



	def mbackward(self):
		self.setTreads(-1,-1)
		time.sleep(.1)
		self.setTreads(0,0)





















	# called by Rover20, acts as main loop
	def processVideo(self, jpegbytes, timestamp_10msec):
		#240,320

		self.currentImage = cv2.imdecode(np.asarray(bytearray(jpegbytes), dtype=np.uint8), 0)
		
		self.pattern = np.tile(np.reshape(self.currentImage,(240*320)),(100,1))

		self.pattern = self.pattern + 0.05*(1-np.random.random((self.pattern.shape[0],self.pattern.shape[1])))

		self.bias=np.ones((self.pattern.shape[0],1))

		self.pattern=np.concatenate((self.pattern,self.bias), axis=1)



		self.act1=np.concatenate((np.squeeze(np.array(self.af(np.dot(1*self.pattern,self.w1)))),self.bias), axis=1)

		self.act2=np.squeeze(np.array(self.af(np.dot(self.act1,self.w2))))



		self.act22=0*self.act2

		for i in range(self.act2.shape[0]):
			self.act22[i,np.argmax(self.act2[i,:])]=1


		a=np.append(self.action_vectors,[np.sum(self.act22,axis=0)/self.act22.shape[0]],axis=0)	

		self.action_choice = input("Enter 1 for left, 2 for forward, 3 for right, 4 for reverse, 5 for network(" 
					+ self.action_labels[np.argmax(np.sum(self.act22,axis=0))]+")")

		self.action_choice=self.action_choice-1

		self.category = np.tile(a[self.action_choice],(self.pattern.shape[0],1))		

		
		if self.action_choice==4:
			self.action_choice=np.argmax(np.sum(self.act22,axis=0))


		if self.action_choice==0:
			self.mleft()
		elif self.action_choice==1:
			self.mforward()
		elif self.action_choice==2:
			self.mright() 
		elif self.action_choice==3:
			self.mbackward()  
		
		

		



		

		

		






		self.error = self.category - self.act2
		self.sse=np.power(self.error,2).sum()

		self.delta_w2=self.error*self.act2*(1-self.act2)
		self.delta_w1=np.dot(self.delta_w2,self.w2.transpose())*self.act1*(1-self.act1)
		self.delta_w1=np.delete(self.delta_w1,-1,1)

		self.dw1=np.dot(self.L,np.dot(self.pattern.transpose(),self.delta_w1))+self.M*self.dw1
		self.dw2=np.dot(self.L,np.dot(self.act1.transpose(),self.delta_w2))+self.M*self.dw2
		self.w1=self.w1+self.dw1
		self.w2=self.w2+self.dw2








		
		





















	























































def main():

	rover = MPCR_Rover()

	while not rover.quit:
		pass

	rover.close()



if __name__ == '__main__':
	main()

