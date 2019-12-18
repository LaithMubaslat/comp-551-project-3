

#The class graph was acquired from geeksforgeeks 
#description of the code 
#1each image is thresholded first (setting values bellow a certain threshold to 0)
#2an incidance matrix is obtained from each image 
#3graph corressponding to that matrix is then obtained 
#4connected components are computeted for the graph 
#5 size of connected components islands is compared 
#island with largest bounding box is returned (rectangule which was found to be insufficient for this work)
#should be made into a square 
#6 indices for that bounding box boundaries are used to return the digit contained inside 



#https://www.geeksforgeeks.org/find-number-of-islands/
#https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/

class Graph: 
      
    # init function to declare class variables 
    def __init__(self,V): 
        self.V = V 
        self.adj = [[] for i in range(V)] 
  
    def DFSUtil(self, temp, v, visited): 
  
        # Mark the current vertex as visited 
        visited[v] = True
  
        # Store the vertex to list 
        temp.append(v) 
  
        # Repeat for all vertices adjacent 
        # to this vertex v 
        for i in self.adj[v]: 
            if visited[i] == False: 
                  
                # Update the list 
                temp = self.DFSUtil(temp, i, visited) 
        return temp 
  
    # method to add an undirected edge 
    def addEdge(self, v, w): 
        self.adj[v].append(w) 
        self.adj[w].append(v) 
  
    # Method to retrieve connected components 
    # in an undirected graph 
    def connectedComponents(self): 
        visited = [] 
        cc = [] 
        for i in range(self.V): 
            visited.append(False) 
        for v in range(self.V): 
            if visited[v] == False: 
                temp = [] 
                cc.append(self.DFSUtil(temp, v, visited)) 
        return cc 


g=Graph(64*64)

def Adjecency (Img):
    Adim=64*64
    A=np.zeros((Adim,Adim))
    Imgindex=np.zeros((64,64))
    G=Graph(64*64)
    
    num=0
    InverseMap=[]
    for i in range (64):
        for j in range (64):
            Imgindex[i][j]=num
            InverseMap.append([i,j])
            num=num+1
    Imgindex=Imgindex.astype(int)       
            
    
    for i in range (64):
        for j in range (64):
            
            if (Img[i][j]>0):
                A_axis_1=Imgindex[i][j]
               
                
                
                #1
                z=i-1
                k=j-1
                if (z>-1 and k>-1):
                    if(Img[z][k]>0):
                        A_axis_2=Imgindex[z][k]
                        G.addEdge(A_axis_1,A_axis_2)
                #2
                z=i-1
                k=j
                if(z>-1):
                    if(Img[z][k]>0):
                        A_axis_2=Imgindex[z][k]
                        G.addEdge(A_axis_1,A_axis_2)
                    
                #3
                z=i-1
                k=j+1
                if (z>-1 and k<64):
                    if(Img[z][k]>0):
                        A_axis_2=Imgindex[z][k]
                        G.addEdge(A_axis_1,A_axis_2)
                    
                
                 #4
                z=i+1
                k=j-1
                if (z<64 and k>-1):
                    if(Img[z][k]>0):
                        A_axis_2=Imgindex[z][k]
                        G.addEdge(A_axis_1,A_axis_2)
                #5
                z=i+1
                k=j
                if (z<64):
                    if(Img[z][k]>0):
                        A_axis_2=Imgindex[z][k]
                        G.addEdge(A_axis_1,A_axis_2)
                #6
                z=i+1
                k=j+1
                if (z<64 and k<64):
                    if(Img[z][k]>0):
                        A_axis_2=Imgindex[z][k]
                        G.addEdge(A_axis_1,A_axis_2)
                    
                
                #7
                z=i
                k=j+1
                if(k<64):
                    if(Img[z][k]>0):
                        A_axis_2=Imgindex[z][k]
                        G.addEdge(A_axis_1,A_axis_2)
                #8
                z=i
                k=j-1
                if (k>-1):
                    if(Img[z][k]>0):
                        A_axis_2=Imgindex[z][k]
                        G.addEdge(A_axis_1,A_axis_2)
                
                
    return G, InverseMap
    
def filter(Img,threshold):
    
    for i in range (len(Img)):
        for j in range (len(Img[0])):
        
            if (Img[i][j]<threshold):
            
                Img[i][j]=0
    
    #for i in range (len(Img)-1):
      #  for j in range (len(Img)-1):
      #      if (Img[i-1][j]==0):
       #        if ( Img[i+1][j]==0):
        #           if(Img[i][j-1]==0):
        #               if(Img[i][j+1]==0):
         #                  Img[i][j]=0
    
    #remove stray points 
    return Img 





        
def seperate(Img): 
    
    
    graph, InverseMap = Adjecency (Img)
            
    cc = graph.connectedComponents()         
            
    connected=[]
    for i in range (len(cc)):
        if (len(cc[i])>1):
            connected.append(cc[i])
            
            
    #get largest and smallest for each 
    
    smallest_i=50000
    smallest_j=50000
    largest_i=0
    largest_j=0
    i_boundary=[]
    j_boundary=[]
    for i in range (len(connected)):
        for j in range (len(connected[i])):
            nodenumber=connected[i][j]
            I=InverseMap[nodenumber][0]
            J=InverseMap[nodenumber][1]
            
            if(I>largest_i):
                largest_i=I
            if(I<smallest_i):
                smallest_i=I
            if(J>largest_j):
                largest_j=J
            if(J<smallest_j):
                smallest_j=J
                 
            
                
                
                
                
        i_boundary.append([smallest_i,largest_i])
        j_boundary.append([smallest_j,largest_j])
        #reset
        smallest_i=50000
        smallest_j=50000
        largest_i=0
        largest_j=0
    
    
    
    seperate_numbers=[]
    for i in range (len(i_boundary)):
        smallest_i=i_boundary[i][0]
        largest_i=i_boundary[i][1]+1
        smallest_j=j_boundary[i][0]
        largest_j=j_boundary[i][1]+1
        image=Img[smallest_i:largest_i,smallest_j:largest_j]
        seperate_numbers.append(image)
    
    return seperate_numbers



#main




def Largest_Number(Img,threshold):
    filtered=filter(Img,threshold)
    seperated= seperate(filtered)
    
    largest=0
    largest_index=0
    for i in range (len(seperated)):
        if (seperated[i].size>largest):
            largest=seperated[i].size
            largest_index=i
    
    count=0
    for i in range (len(seperated)):
        if (largest==seperated[i].size):
            count=count+1
    
    if(count==2):
        return np.zeros((1,1))
            
    Largest_num=seperated[largest_index]
    
    return  Largest_num


i=1000
img_idx=i
Img= train_images[img_idx] 
plt.imshow(Img)
Num=Largest_Number(Img,threshold)
plt.imshow(Num)

threshold=200










#plt.imshow(seperated[0])
#plt.imshow(seperated[1])
#plt.imshow(seperated[2])

#plt.imshow(Img)