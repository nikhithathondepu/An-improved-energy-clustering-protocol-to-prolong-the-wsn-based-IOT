import tkinter
from tkinter import *
import math
import random
from threading import Thread 
from collections import defaultdict
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from fcmeans import FCM
import ModifiedFCM
from ModifiedFCM import *

global mobile, labels, mobile_x, mobile_y, text, canvas, mobile_list, root, num_nodes, tf1, nodes, optimal_cluster, cluster_label, ieeecp, eecp
option = 0
energy = {}
cluster_head = []
colors = ['red', 'green', 'blue', 'yellow', 'pink', 'orange', 'magenta', 'cyan', 'snow', 'lavender', 'OliveDrab1', 'OliveDrab2', 'OliveDrab4',
          'DarkOliveGreen1', 'DarkOliveGreen2', 'DarkOliveGreen3', 'DarkOliveGreen4', 'khaki1', 'khaki2', 'khaki3', 'khaki4']

M = 300 #WSN area size
Pe = 0.85 #Pe threshold

def getDistance(iot_x,iot_y,x1,y1):
    flag = False
    for i in range(len(iot_x)):
        dist = math.sqrt((iot_x[i] - x1)**2 + (iot_y[i] - y1)**2)
        if dist < 60:
            flag = True
            break
    return flag

def generateWSN():
    global mobile, labels, mobile_x, mobile_y, num_nodes, tf1, nodes
    mobile = []
    mobile_x = []
    mobile_y = []
    labels = []
    nodes = []
    canvas.update()
    num_nodes = int(tf1.get().strip())
    x = 5
    y = 350
    mobile_x.append(x)
    mobile_y.append(y)
    name = canvas.create_oval(x,y,x+40,y+40, fill="blue")
    lbl = canvas.create_text(x+20,y-10,fill="darkblue",font="Times 7 italic bold",text="BS")
    labels.append(lbl)
    mobile.append(name)

    for i in range(1,num_nodes):
        run = True
        while run == True:
            x = random.randint(100, 450)
            y = random.randint(50, 600)
            flag = getDistance(mobile_x,mobile_y,x,y)
            if flag == False:
                nodes.append([x, y])
                mobile_x.append(x)
                mobile_y.append(y)
                run = False
                name = canvas.create_oval(x,y,x+40,y+40, fill="red")
                lbl = canvas.create_text(x+20,y-10,fill="darkblue",font="Times 8 italic bold",text="MN "+str(i))
                labels.append(lbl)
                mobile.append(name)    

def aodvTransfer(text,canvas,line1,line2,x1,y1,x2,y2):
    class AODVSimulationThread(Thread):
        def __init__(self,text,canvas,line1,line2,x1,y1,x2,y2): 
            Thread.__init__(self) 
            self.canvas = canvas
            self.line1 = line1
            self.line2 = line2
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.text = text            
 
        def run(self):
            time.sleep(1)
            for i in range(0,3):
                self.canvas.delete(self.line1)
                self.canvas.delete(self.line2)
                time.sleep(1)
                self.line1 = canvas.create_line(self.x1, self.y1,self.x2, self.y2,fill='black',width=3)
                self.line2 = canvas.create_line(self.x2, self.y2, 25, 370,fill='black',width=3)
                time.sleep(1)               
            self.canvas.delete(self.line1)
            self.canvas.delete(self.line2)
            canvas.update()
                
    newthread = AODVSimulationThread(text,canvas,line1,line2,x1,y1,x2,y2) 
    newthread.start() 

def existingAODV():
    selected_node = None
    distance = 10000
    src = int(mobile_list.get()) - 1
    for i in range(1,20):
        x1 = mobile_x[i]
        y1 = mobile_y[i]
        dist = math.sqrt((5 - x1)**2 + (350 - y1)**2)
        print(str(i)+" "+str(dist))
        if dist < distance:
            distance = dist
            selected_node = i - 1           
    print(str(selected_node)+" === "+str(src))    
    temp = nodes[src]
    src_x = temp[0]
    src_y = temp[1]
    temp = nodes[selected_node]
    cls_x = temp[0]
    cls_y = temp[1]    
    line1 = canvas.create_line(src_x+20, src_y+20,cls_x+20, cls_y+20,fill='black',width=3)
    line2 = canvas.create_line(cls_x+20, cls_y+20, 25, 370,fill='black',width=3)
    aodvTransfer(text,canvas,line1,line2,(src_x+20),(src_y+20),(cls_x+20),(cls_y+20))
    option = 1    

def findoptimalCluster():
    global optimal_cluster
    text.delete('1.0', END)
    #base station location and Mobile node location
    x1 = 5 #base station X position
    y1 = 350 #base station Y position
    x2 = 450 #WSN width area size
    y2 = 600 #WSN height area size

    dist_bs = math.sqrt((x1 - x2)**2 + (y1 - y2)**2) #get distances between base station and WSN area

    optimal_cluster = math.sqrt((1.262 * num_nodes) / 2 * math.pi) #get optimal number of clusters
    optimal_cluster = int(optimal_cluster * (M / dist_bs))
    text.insert(END,"Optimal Number of Clusters = "+str(optimal_cluster)+"\n\n")

def modifiedFCM():
    text.delete('1.0', END)
    global nodes, optimal_cluster, cluster_label, mobile, labels, num_nodes
    nodes = np.asarray(nodes)
    fcm = FCM(n_clusters = optimal_cluster)#create traditional FCM cluster
    fcm.fit(nodes) #now train and get cluster label from existing FCM
    cluster_label = fcm.predict(nodes)
    centers = fcm.centers
    cluster_th = (num_nodes * Pe) / optimal_cluster #calculate cluster threshold
    if num_nodes > 25 and optimal_cluster < cluster_th: #if cluster optimal size < cluster threshold then call Modified FCM to create clusters based on nearest neigbors
        cluster_label, selected_cent = ModifiedFCM.modifiedFCM(nodes, 50, optimal_cluster, centers) #here calling modified FCM to rearrnage cluster
    for i in range(len(nodes)): #now looping and printing all nodes with its cluster id
        text.insert(END,"Node id : "+str(i)+" is in cluster : "+str(cluster_label[i])+"\n")
    canvas.update()    
    for i in range(1, len(labels)):
        canvas.delete(labels[i])
        canvas.delete(mobile[i])
    canvas.update()
    for i in range(len(nodes)):
        node = nodes[i]
        name = canvas.create_oval(node[0],node[1],node[0]+40,node[1]+40, fill=colors[cluster_label[i]])
        lbl = canvas.create_text(node[0]+20,node[1]-10,fill="black",font="Times 8 italic bold",text="MN "+str(i))
        labels[i+1] = lbl
        mobile[i+1] = name
    canvas.update()    
    
def getClusterHead(data):
    cluster_id = None
    global cluster_head
    for k in range(len(data)):
        d = data[k]
        if d[0] in cluster_head:
            cluster_id = d[0]
            break
    return cluster_id        

def CHSelection():
    text.delete('1.0', END)
    global cluster_label, energy, cluster_head
    energy.clear()
    cluster_head.clear()
    for i in range(len(cluster_label)):
        en = random.randint(10, 100)
        if cluster_label[i] in energy.keys():
            energy[cluster_label[i]].append([i, en])
        else:
            temp = []
            temp.append([i, en])
            energy[cluster_label[i]] = temp
    for key, value in energy.items(): #loop all nodes and its available energy
        selected_ch = None
        en = 0
        for i in range(len(value)):#get each node from the cluster
            data = value[i]
            if data[1] > en: #look for node with high battery
                en = data[1]
                selected_ch = data[0]
        cluster_head.append(selected_ch)
    for i in range(len(cluster_head)):
        text.insert(END,"Selected Cluster Head for Cluster "+str(i)+" is "+str(cluster_head[i])+"\n")

    canvas.update()    
    for i in range(1,len(labels)):
        canvas.delete(labels[i])
        canvas.delete(mobile[i])
    canvas.update()
    for i in range(len(nodes)):
        node = nodes[i]
        name = canvas.create_oval(node[0],node[1],node[0]+40,node[1]+40, fill=colors[cluster_label[i]])
        if i in cluster_head:
            lbl = canvas.create_text(node[0]+20,node[1]-10,fill="black",font="Times 8 italic bold",text="CH "+str(i))
        else :
            lbl = canvas.create_text(node[0]+20,node[1]-10,fill="black",font="Times 8 italic bold",text="MN "+str(i))    
        labels[i+1] = lbl
        mobile[i+1] = name
    canvas.update()

def startDataTransferSimulation(text,canvas,line1,line2,line3,x1,y1,x2,y2,x3,y3):
    class SimulationThread(Thread):
        def __init__(self,text,canvas,line1,line2,line3,x1,y1,x2,y2,x3,y3): 
            Thread.__init__(self) 
            self.canvas = canvas
            self.line1 = line1
            self.line2 = line2
            self.line3 = line3
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.x3 = x3
            self.y3 = y3
            self.text = text            
 
        def run(self):
            time.sleep(1)
            for i in range(0,3):
                self.canvas.delete(self.line1)
                self.canvas.delete(self.line2)
                self.canvas.delete(self.line3)
                time.sleep(1)
                self.line1 = canvas.create_line(self.x1, self.y1,self.x2, self.y2,fill='black',width=3)
                self.line2 = canvas.create_line(self.x2, self.y2,self.x3, self.y3,fill='black',width=3)
                self.line3 = canvas.create_line(self.x3, self.y3,25, 370,fill='black',width=3)
                time.sleep(1)               
            self.canvas.delete(self.line1)
            self.canvas.delete(self.line2)
            self.canvas.delete(self.line3)
            canvas.update()
                
    newthread = SimulationThread(text,canvas,line1,line2,line3,x1,y1,x2,y2,x3,y3) 
    newthread.start()    

def sendPacket():
    text.delete('1.0', END)
    global energy, ieeecp, eecp
    ieeecp = 0
    eecp = 0
    src = int(mobile_list.get())
    cluster_id = None
    for key, value in energy.items():
        for i in range(len(value)):
            data = value[i]
            if data[0] == src:
                cluster_id = getClusterHead(value)
                break
    text.insert(END,"Selected Source : "+str(src)+"\n")
    text.insert(END,"Cluster Head for Selected Source : "+str(cluster_id)+"\n")
    temp = nodes[src]
    src_x = temp[0]
    src_y = temp[1]
    temp = nodes[cluster_id]
    cls_x = temp[0]
    cls_y = temp[1]
    src = src - 1
    #cluster_id = cluster_id - 1
    hop = -1
    distance = 10000
    distance1 = 10000
    for i in range(len(nodes)):
        node = nodes[i]
        temp_x = node[0]
        temp_y = node[1]
        if i != src and i != cluster_id and temp_x < cls_x:
            dist = math.sqrt((cls_x - temp_x)**2 + (cls_y - temp_y)**2)
            dist1 = math.sqrt((5 - temp_x)**2 + (350 - temp_y)**2)
            if dist < distance and dist1 < distance1:
                distance = dist
                distance1 = dist1
                hop = i
                ieeecp = ieeecp + 1
                eecp = eecp + 1.5
    if hop != -1:
        ieeecp = ieeecp / num_nodes
        eecp = eecp / num_nodes
        temp = nodes[hop]
        hop_x = temp[0]
        hop_y = temp[1]
        text.insert(END,"Selected Nearest Neighbor = MN"+str(hop)+"\n")
        line1 = canvas.create_line(src_x+20, src_y+20,cls_x+20, cls_y+20,fill='black',width=3)
        line2 = canvas.create_line(cls_x+20, cls_y+20,hop_x+20, hop_y+20,fill='black',width=3)
        line3 = canvas.create_line(hop_x+20, hop_y+20,25, 370,fill='black',width=3)
        startDataTransferSimulation(text,canvas,line1,line2,line3,(src_x+20),(src_y+20),(cls_x+20),(cls_y+20),(hop_x+20),(hop_y+20))
        option = 1
    else:
        text.insert(END,"Unable to find path. please choose other node\n")

def graph():
    global ieeecp, eecp
    height = [ieeecp, eecp]
    bars = ('Propose IEEECP Energy','Existing EECP Energy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title('Energy Consumption Comparison Graph')
    plt.xlabel('Technique Name')
    plt.ylabel('Energy Consumption')
    plt.show()

def Main():
    global root, tf1, text, canvas, mobile_list
    root = tkinter.Tk()
    root.geometry("1300x1200")
    root.title("An Improved Energy-Efficient Clustering Protocol to Prolong the Lifetime of the WSN-Based IoT")
    root.resizable(True,True)
    font1 = ('times', 12, 'bold')

    canvas = Canvas(root, width = 800, height = 700)
    canvas.pack()

    l2 = Label(root, text='Num Nodes:')
    l2.config(font=font1)
    l2.place(x=820,y=10)

    tf1 = Entry(root,width=10)
    tf1.config(font=font1)
    tf1.place(x=970,y=10)

    l1 = Label(root, text='Node ID:')
    l1.config(font=font1)
    l1.place(x=820,y=60)

    mid = []
    for i in range(1,100):
        mid.append(str(i))
    mobile_list = ttk.Combobox(root,values=mid,postcommand=lambda: mobile_list.configure(values=mid))
    mobile_list.place(x=970,y=60)
    mobile_list.current(0)
    mobile_list.config(font=font1)

    createButton = Button(root, text="Generate WSN Network", command=generateWSN)
    createButton.place(x=820,y=110)
    createButton.config(font=font1)

    existingButton = Button(root, text="Existing AODV Routing Algorithm", command=existingAODV)
    existingButton.place(x=1020,y=110)
    existingButton.config(font=font1)

    optimalButton = Button(root, text="Find Optimal Cluster Size", command=findoptimalCluster)
    optimalButton.place(x=820,y=160)
    optimalButton.config(font=font1)

    mfcmButton = Button(root, text="Run Modified FCM", command=modifiedFCM)
    mfcmButton.place(x=820,y=210)
    mfcmButton.config(font=font1)

    chselectButton = Button(root, text="Rotation Cluster Head Selection", command=CHSelection)
    chselectButton.place(x=820,y=260)
    chselectButton.config(font=font1)

    sendpacketButton = Button(root, text="Route Packets to Base Station", command=sendPacket)
    sendpacketButton.place(x=820,y=310)
    sendpacketButton.config(font=font1)

    graphButton = Button(root, text="Energy Consumption Graph", command=graph)
    graphButton.place(x=1060,y=310)
    graphButton.config(font=font1)

    text=Text(root,height=18,width=60)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=820,y=360)
    
    
    root.mainloop()
   
 
if __name__== '__main__' :
    Main ()
    
