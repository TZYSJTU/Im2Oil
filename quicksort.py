def partition(arr,low,high): 
    i = ( low-1 )         # 最小元素索引
    pivot = arr[high]     
  
    for j in range(low , high): 
  
        # 当前元素小于或等于 pivot 
        if   arr[j]['importance'] >= pivot['importance']: 
          
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 
  
 
# arr[] --> 排序数组
# low  --> 起始索引
# high  --> 结束索引
  
# 快速排序函数
def quickSort(arr,low,high): # 大的在前
    if low < high: 
  
        pi = partition(arr,low,high) 
  
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high) 
  


# print ("排序后的数组:") 
# for i in range(n): 
#     print ("%d" %arr[i]),



def partition_list(arr,low,high): 
    i = (low-1)         # 最小元素索引
    pivot = arr[high]     
  
    for j in range(low , high): 
  
        # 当前元素小于或等于 pivot 
        if   arr[j][2] >= pivot[2]: 
          
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 

def quickSort_list(arr,low,high): # 大的在前
    if low < high: 
  
        pi = partition_list(arr,low,high) 
  
        quickSort_list(arr, low, pi-1) 
        quickSort_list(arr, pi+1, high) 


#################### 
# for render sort
def partition_stroke(arr, index, low,high): 
    i = (low-1)         # 最小元素索引
    pivot = arr[high]     
  
    for j in range(low , high): 
  
        # 当前元素小于或等于 pivot 
        if   arr[j] <= pivot: 
          
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
            index[i],index[j] = index[j],index[i] 
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    index[i+1],index[high] = index[high],index[i+1] 
    return ( i+1 ) 

def quickSort_stroke(arr,index,low,high): # 小的在前
    if low < high: 
  
        pi = partition_stroke(arr,index,low,high) 
  
        quickSort_stroke(arr, index, low, pi-1) 
        quickSort_stroke(arr, index, pi+1, high) 
    return index

####### for stroke_group ######
def partition_group(arr,low,high): 
    i = (low-1)         # 最小元素索引
    pivot = arr[high]     
  
    for j in range(low , high): 
  
        # 当前元素小于或等于 pivot 
        if   arr[j][-1] >= pivot[-1]: 
          
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return ( i+1 ) 

def quickSort_group(arr,low,high): # 大的在前
    if low < high: 
  
        pi = partition_group(arr,low,high) 
  
        quickSort_group(arr, low, pi-1) 
        quickSort_group(arr, pi+1, high) 