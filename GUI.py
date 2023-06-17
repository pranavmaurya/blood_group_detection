from tkinter import *
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib import pyplot as plt
#from metrics import score
import numpy as np # used for handling numbers
import pandas as pd
from sklearn.preprocessing import Normalizer

from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling
import numpy as np
import os

from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

blood=[False,False,False,False]
q = 1
f = 0
v = 0
p1 = ''
p2 = ''
p3 = ''
p4 = ''
fname=''


class Login(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):

        # Code Segment for Icon and title

        self.configure(background='powder blue')
        self.pack(fill=BOTH, expand=1)
        self.master.title("Blood Group Detection System")
        self.master.iconbitmap('Blood.ico')

        # Code segment for dropdown menu

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        stage = Menu(menu)
        file.add_cascade(label="Process", menu=stage)
        file.add_command(label="Restart",command=self.restart)
        file.add_command(label="Exit",command=self.quit)
        menu.add_cascade(label="File", menu=file)

        stage.add_command(label="Process 1: Green Plane Extraction", command=self.gp)
        stage.add_command(label="Process 2: Auto Threshold", command=self.autothresh)
        stage.add_command(label="Process 3: Adaptive Threshold:Ni Black", command=self.Adapthresh)
        stage.add_command(label="Process 4: Morphology: Fill Holes", command=self.Fill_holes)
        stage.add_command(label="Process 5: Advanced Morphology: Remove small objects", command=self.Remove_small_objects)
        stage.add_command(label="Process 6: Histogram", command=self.Histogram)
        stage.add_command(label="Process 7: Quantification", command=self.HSV_Luminance)

        # Code segment for labels
        l1 = Label(self, text="Reagent Anti-A", font=("Helvetica", 12))
        l2 = Label(self, text="Reagent Anti-B", font=("Helvetica", 12))
        l3 = Label(self, text="Reagent Anti-D", font=("Helvetica", 12))
        l4 = Label(self, text="Reagent Anti-H", font=("Helvetica", 12))
        l1.place(x=250, y=780)
        l2.place(x=560, y=780)
        l3.place(x=860, y=780)
        l4.place(x=1160, y=780)

        
        ## Code segment for buttons
        #e1 = Button(self, text="Choose Image", command=self.imagesel1)
        #e2 = Button(self, text="Choose Image", command=self.imagesel2)
        #e3 = Button(self, text="Choose Image", command=self.imagesel3)
        #e4 = Button(self, text="Choose Image", command=self.imagesel4)
        #self.ep = Button(self, text="Process", font=("Helvetica", 12), fg='red', relief=SUNKEN)
        #self.ep.place(x=650, y=710)
        #e1.place(x=170, y=500)
        #e2.place(x=490, y=500)
        #e3.place(x=790, y=500)
        #e4.place(x=1080, y=500)

        e5=Button(self, text="Choose Image", command=self.imgload)
        e5.place(x=10,y=120)
        l5 = Label(self, text="Blood group detector", fg="red", font=("Times New Roman", 23))
        l5.place(x=630, y=15)
        #usernameLabel = Label(tkWindow, text="User Name").grid(row=0, column=0)
        #username = StringVar()
        #usernameEntry = Entry(tkWindow, textvariable=username).grid(row=0, column=1)  
        
    def quit(self):
        global q
        q = 0
        root.destroy()


    def restart(self):
        global q
        q = 1
        root.destroy()

    def  message(self,q):
        messagebox.showinfo("Result",q+"Confirmed")

    def start1(self):
        self.start(p1,"Anti A")
        self.start2()

    
    '''def imagesel1(self):
        global v
        v += 1
        s = filedialog.askopenfilename()
        x = ""
        i = len(s)-1
        while s[i] != '/':
            x += s[i]
            i -= 1
        global p1
        p1 = x[::-1]
        self.p = Image.open(x[::-1])
        r = self.p.resize((300,425),Image.ANTIALIAS)
        i = ImageTk.PhotoImage(r)
        l = Label(self, image=i)
        l.Image = i
        l.place(x=75, y=50)
        if v == 4:
            self.ep.configure(relief=RAISED, fg='green', command=self.start1)

    def start2(self):
        self.start(p2, "Anti B")
        self.start3()



    def imagesel2(self):
        global v, p2
        v += 1
        s = filedialog.askopenfilename()
        x = ""
        i = len(s)-1
        while s[i] != '/':
            x += s[i]
            i -= 1
        p2 = x[::-1]
        self.p = Image.open(x[::-1])
        r = self.p.resize((300, 425), Image.ANTIALIAS)
        i = ImageTk.PhotoImage(r)
        l = Label(self, image=i)
        l.Image = i
        l.place(x=375, y=50)
        if v == 4:
            self.ep.configure(relief=RAISED, fg='green', command=self.start1)

    def start3(self):
        self.start(p3, "Anti D")
        self.start4()


    def imagesel3(self):
        global v, p3
        v += 1
        s = filedialog.askopenfilename()
        x = ""
        i = len(s)-1
        while s[i] != '/':
            x += s[i]
            i -= 1
        p3 = x[::-1]
        self.p = Image.open(x[::-1])
        r = self.p.resize((300,425),Image.ANTIALIAS)
        i = ImageTk.PhotoImage(r)
        l = Label(self, image=i)
        l.Image = i
        l.place(x=675, y=50)
        if v == 4:
            self.ep.configure(relief=RAISED, fg='green', command=self.start1)

    def start4(self):
        self.start(p4, "Control")
        self.check()

    def imagesel4(self):
        global v, p4
        v += 1
        s = filedialog.askopenfilename()
        x = ""
        i = len(s)-1
        while s[i] != '/':
            x += s[i]
            i -= 1
        p4 = x[::-1]
        self.p = Image.open(x[::-1])
        r = self.p.resize((300, 425), Image.ANTIALIAS)
        i = ImageTk.PhotoImage(r)
        l = Label(self, image=i)
        l.Image = i
        l.place(x=975, y=50)
        if v == 4:
            self.ep.configure(relief=RAISED, fg='green', command=self.start1)'''
        
    

    

    def imgload(self):
        #global fname
        #s =  filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("CSV files","*.jpg"),("all files","*.*")))
        #print (fname)

        print("Image copied to environment")
        global blood        
        global q         
        global f       
        global v       
        global p1       
        global p2    
        global p3      
        global p4    
        global fname
        global pid
        

        blood=[False,False,False,False]
        q = 1
        f = 0
        v = 0
        p1 = ''
        p2 = ''
        p3 = ''
        p4 = ''
        fname=''
        
        global pid
        global pname
        #global v
        v += 1
        s = filedialog.askopenfilename()
        x = ""
        #print(s)
        i = len(s)-1
        #print(i)
        while s[i] != '/':
            x += s[i]
            #print(x)
            i -= 1
            #print(i)
        #global p1

        
        p = x[::-1]
        self.p = Image.open(x[::-1])
        r = self.p.resize((500,225),Image.ANTIALIAS)
        i = ImageTk.PhotoImage(r)
        l = Label(self, image=i)
        l.Image = i
        l.place(x=165, y=80)
        print(" ")
        
        irt=1
        if irt==1:
            #print("aaaaaaaaaaaaaaaaaaa")
            
            import time
            start_time = time.time()
            img = cv2.imread(s)
            
            #img=cv2.resize(img, (300, 300))

            #from here the single image gets split and place in the respective place
            x=0
            y=0        
            h=img.shape[0]
            w=274
            crop_img = img[y:y+h, x:x+w]
            #cv2.imshow("cropped", crop_img)
            #cv2.waitKey()
            im1name="A1.jpg"
            cv2.imwrite(im1name, crop_img) 
            p1 = im1name#x[::-1]
            self.p = Image.open(im1name)
            r = self.p.resize((300,425),Image.ANTIALIAS)
            i = ImageTk.PhotoImage(r)
            l = Label(self, image=i)
            l.Image = i
            l.place(x=165, y=330)
            print(i)
            #print(im1name)
            #if v == 4:
                #self.ep.configure(relief=RAISED, fg='green', command=self.start1)


                


            x=300
            y=0        
            h=img.shape[0]
            w=274
            crop_img = img[y:y+h, x:x+w]
            #cv2.imshow("cropped", crop_img)
            #cv2.waitKey()
            im1name="A2.jpg"
            cv2.imwrite(im1name, crop_img)
            
            #global p2
            p2 = im1name#x[::-1]
            self.p = Image.open(im1name)
            r = self.p.resize((300,425),Image.ANTIALIAS)
            i = ImageTk.PhotoImage(r)
            l = Label(self, image=i)
            l.Image = i
            l.place(x=465, y=330)
            
            #if v == 4:
                #self.ep.configure(relief=RAISED, fg='green', command=self.start1)


            x=590
            y=0        
            h=img.shape[0]
            w=274
            crop_img = img[y:y+h, x:x+w]
            #cv2.imshow("cropped", crop_img)
            #cv2.waitKey()
            im1name="A3.jpg"
            cv2.imwrite(im1name, crop_img)
            
            #global p3
            p3 = im1name#x[::-1]
            self.p = Image.open(im1name)
            r = self.p.resize((300,425),Image.ANTIALIAS)
            i = ImageTk.PhotoImage(r)
            l = Label(self, image=i)
            l.Image = i
            l.place(x=765, y=330)
            
            #if v == 4:
                #self.ep.configure(relief=RAISED, fg='green', command=self.start1)





            x=890
            y=0        
            h=img.shape[0]
            w=274
            crop_img = img[y:y+h, x:x+w]
            #cv2.imshow("cropped", crop_img)
            #cv2.waitKey()
            im1name="A4.jpg"
            cv2.imwrite(im1name, crop_img)
            
            p4 = im1name#x[::-1]
            self.p = Image.open(im1name)
            r = self.p.resize((300,425),Image.ANTIALIAS)
            i = ImageTk.PhotoImage(r)
            l = Label(self, image=i)
            l.Image = i
            l.place(x=1065, y=330)
            
            #if v == 4:
            #self.ep.configure(relief=RAISED, fg='green', command=self.start1)

            self.start(p1,"Anti A")
            self.start(p2, "Anti B")
            self.start(p3, "Anti D")        
            self.start(p4, "Anti H")
            self.check()



            Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
                    'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]       }

            #print(Data)

            #df=pd.DataFrame({'x':Data.,'y':X })
                     

          
            df = DataFrame(Data,columns=['x','y'])
            #print(df)
            #X=data[4]
            #Y=data[5]


            #print(data.iloc[23000])
            #print(data.iloc[46000])
            kmeans = KMeans(n_clusters=4).fit(df)
            centroids = kmeans.cluster_centers_
            print(centroids)
            print("---Time Taken is %s seconds ---" % (time.time() - start_time))
                                       
            plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
            #plt.scatter(X, Y, c= kmeans.labels_.astype(float), s=50, alpha=0.5)
            plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
            plt.title('K-Means Cluster')
            plt.show()

            
            




























            

    def process1(self, p,r):  # Extracting the Green plane
        img = cv2.imread(p)
        gi = img[:, :, 1]
        cv2.imwrite("p1"+r+".png", gi)
        return gi

    def process2(self, p,r):  # Obtaining the threshold
        gi = self.process1(p,r)
        _, th = cv2.threshold(gi, 0, 255, cv2.THRESH_OTSU)
        cv2.imwrite("p2"+r+".png", th)

    def process3(self, p,r):  # Obtaining Ni black image
        img = cv2.imread('p2'+r+'.png', 0)
        th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 14)
        cv2.imwrite("p3"+r+".png", th4)

    def process4(self,r):  # Morphology: fill holes
        gi = cv2.imread('p3'+r+'.png', cv2.IMREAD_GRAYSCALE)
        th, gi_th = cv2.threshold(gi, 220, 255, cv2.THRESH_BINARY_INV)
        gi_floodFill=gi_th.copy()
        h, w = gi_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(gi_floodFill, mask, (0, 0), 255)
        gi_floodFill_inv = cv2.bitwise_not(gi_floodFill)
        gi_out = gi_th | gi_floodFill_inv
        cv2.imwrite('p4'+r+'.png', gi_out)

    def process5(self,r):  # Morphing To eliminate small objects
        img = cv2.imread('p4'+r+'.png')
        kernel = np.ones((5, 5), np.uint8)
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('p5'+r+'.png', close)

    def process7(self,r):  #Histogram
        img = cv2.imread('p5'+r+'.png', 0)
        img2 = cv2.imread('p1'+r+'.png', 0)
        mask = np.ones(img.shape[:2], np.uint8)
        hist = cv2.calcHist([img2], [0], mask, [256], [0, 256])
        min = 1000
        max = 0
        n = 0
        s = 0
        ss = 0
        for x, y in enumerate(hist):
            if y > max:
                max = y
            if y < min:
                min = y
            s += y
            n += 1

        mean = s/n
        for x, y in enumerate(hist):
            ss += (y-mean)**2
        ss /= n
        sd = abs(ss)**0.5
        print(r,"-",sd,"\n")
        if sd < 580:
            return 1
        else:
            return 0


    def start(self, p,r):
        global blood
        print(p)
        self.process1(p,r)
        self.process2(p,r)
        self.process3(p,r)
        self.process4(r)
        self.process5(r)
        a = self.process7(r)
        print(a," - ",r)
        if a == 1:
            if r == "Anti A":
                blood[0]=True
            elif r == "Anti B":
                blood[1]=True
            elif r == "Anti D":
                blood[2]=True
            elif r == "Control":
                blood[3]=True

    def check(self):
        if blood[0] is True and blood[1] is True and blood[2] is False and blood[3] is True:
            self.message("Bombay blood")
        elif blood[0] is False and blood[1] is False and blood[2] is True and blood[3] is False:
            self.message("O+")
        elif blood[0] is False and blood[1] is False and blood[2] is False and blood[3] is False:
            self.message("O-")
        elif blood[0] is True and blood[1] is False and blood[2] is True and blood[3] is False:
            self.message("A+")
        elif blood[0] is True and blood[1] is False and blood[2] is False and blood[3] is False:
            self.message("A-")
        elif blood[0] is False and blood[1] is True and blood[2] is True and blood[3] is False:
            self.message("B+")
        elif blood[0] is False and blood[1] is True and blood[2] is False and blood[3] is False:
            self.message("B-")
        elif blood[0] is True and blood[1] is True and blood[2] is True and blood[3] is False:
            self.message("AB+")
        elif blood[0] is True and blood[1] is True and blood[2] is False and blood[3] is False:
            self.message("AB-")


    def gp(self):
        im1 = cv2.imread('p1Anti A.png')
        cv2.imshow('Anti-A',im1)
        im2 = cv2.imread('p1Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p1Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p1Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def autothresh(self):
        im1 = cv2.imread('p2Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p2Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p2Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p2Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Adapthresh(self):
        im1 = cv2.imread('p3Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p3Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p3Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p3Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Fill_holes(self):
        im1 = cv2.imread('p4Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p4Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p4Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p4Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Remove_small_objects(self):
        im1 = cv2.imread('p5Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p5Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p5Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p5Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Histogram(self):
        img1 = cv2.imread('p5Anti A.png', 0)
        img2 = cv2.imread('p5Anti B.png', 0)
        img3 = cv2.imread('p5Anti D.png', 0)
        img4 = cv2.imread('p5Control.png', 0)
        plt.hist(img1.ravel(), 256, [0, 256])
        plt.show()
        plt.hist(img2.ravel(), 256, [0, 256])
        plt.show()
        plt.hist(img3.ravel(), 256, [0, 256])
        plt.show()
        plt.hist(img4.ravel(), 256, [0, 256])
        plt.show()

    def HSV_Luminance(self):
        img1 = cv2.imread(p1)
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv1, 0)

        img2 = cv2.imread(p2)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv2, 0)

        img3 = cv2.imread(p3)
        hsv3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv3, 0)

        img4 = cv2.imread(p4)
        hsv4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv4, 0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def stp_full(self,event=None):
        root.attributes("-fullscreen", False)
        root.geometry("1020x720")




while(1):
    if q == 0:
        break
    else:
        root = Tk()
        root.attributes("-fullscreen",True)
        app = Login(root)
        root.bind("<Escape>", app.stp_full)
        root.mainloop()
