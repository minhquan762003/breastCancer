# Tìm tập phổ biến bằng thuật toán FP-Growth

from breastCancer import df_filled
import numpy as np
from collections import OrderedDict
import csv

class fpTreeNode:
    def __init__(self, name, count, parent):
        self.name = name # tên phần tử
        self.count = count # tần suất (đếm)
        self.parent = parent # nút cha
        self.child = OrderedDict() # nút con
        self.link = None # liên kết đến các nút con cùng tên
       
    # Hiển thị cây dưới dạng danh sách lồng nhau
    def display_tree_list(self):
        print(self.name, self.count,end='')
        if len(self.child)>0:
            print(",[",end='')
        for c in self.child.values():
            print("[",end='')
            c.display_tree_list()
            if len(c.child)==0:
                print("]",end='')
        print("]",end='')


# xuất tập phổ biến vào file csv
def export_to_file(data):
    with open(output_file_name, "w",  newline='') as f:
        writer = csv.writer(f, delimiter=',')
        for row in data:
            writer.writerows([[row]])


# liên kết từ nút hiện tại với nút trước đó
def header_item_table_update(header_item, present_node):
    while (header_item.link != None):
        header_item = header_item.link
    header_item.link = present_node


"""đọc dữ liệu từ file csv.
   thiết lập bảng độ phổ biến của các phần tử
   loại bỏ những phần tử có tần suất < minsup """
def fp_tree_preprocess(doc_name, minsup):
    data = np.genfromtxt(doc_name, delimiter=',', dtype=str)#file_delimiter
    print("data: ",data)
    item_count = {}# lưu bảng độ phổ biến của từng mục (phần tử) trong dữ liệu
    #duyệt CSDL lần 1
    for (x,y), value in np.ndenumerate(data):# hàm ndenumerate thêm vào bộ đếm trước mỗi mục
        if value != '':
            if value not in item_count:
                item_count[value] = 1
            else:
                item_count[value] += 1
    # loại bỏ những mục có tần suất < độ hỗ trợ tối thiểu minsup
    item_count = {k:v for k,v in item_count.items() if v >= minsup}
    print("")
    print("item_count: ",item_count)
    print("")
    return data, item_count

"""sắp xếp lại các mục trong mỗi bản ghi theo bảng tần suất.
    gọi hàm tạo FPtree cho dữ liệu đã được sắp xếp"""
def fp_tree_reorder(data, item_count):
    root = fpTreeNode('Root',1,None)
    #Sắp xếp các mục theo tần suất giảm dần
    #nếu tần suất trùng nhau thì sắp theo bảng chữ cái
    sorted_item_count = sorted(item_count.items(), key=lambda x: (-x[1],x[0]))
    # thiết lập bảng chỉ mục
    sorted_keys = []
    header_item_dict = {}
    for key in sorted_item_count:
        header_item_dict[key[0]] = None 
        sorted_keys.append(key[0]) # danh sách các mục đã được sắp xếp
 
    print("")
    print("header_item_dict ", header_item_dict)
    print("")
    print("sorted_keys ", sorted_keys)
    print("")
    # duyệt từng bản ghi trong data
    for row in data:
        # loại bỏ những mục có tần suất < minsup cho vào trong trans
        trans = []
        for col in row:
            if col in item_count:
                trans.append(col)
                
        # sắp xếp các mục của bản ghi theo danh sach sắp xếp.
        ordered_trans = []
        for item in sorted_keys:
            if item in trans:
                ordered_trans.append(item)
        # cập nhật lên cây fptree
        if len(ordered_trans)!= 0:
            fp_tree_create_and_update(root, ordered_trans, header_item_dict)
    
    print("header_item_dict ", header_item_dict)
    print("")
    return root, header_item_dict


"""hàm đệ quy tạo cây cho mỗi bản ghi."""
def fp_tree_create_and_update(init_node, trans, header_item_dict):
    # nếu nút đã tồn tại thì tăng tần suất 
    if trans[0] in init_node.child:
        init_node.child[trans[0]].count += 1    
    else:
        init_node.child[trans[0]] = fpTreeNode(trans[0], 1, init_node)
        # nút mới được cập nhật thì bảng header cũng được cập nhật
        if header_item_dict[trans[0]] == None:
            header_item_dict[trans[0]] = init_node.child[trans[0]]
        else:
        #dịch chuyển đến nút cuối cùng và cập nhật nút mới
            header_item_table_update(header_item_dict[trans[0]],init_node.child[trans[0]])
    # The function is recursively called for every item in a transaction
    if len(trans) > 1:
        fp_tree_create_and_update(init_node.child[trans[0]], trans[1::],header_item_dict)


"""Hàm để vẽ những cây fptree cho tập phổ biến (tương tự fptree)"""
def conditional_fptree(name,init_node,data):
    if data[0][0] == name:
        # bỏ qua nếu dữ liệu chỉ có một mục
        if len(data)>1:
            conditional_fptree(name,init_node,data[1::])
    if data[0][0] != name:
        if data[0][0] in init_node.child:
            init_node.child[data[0][0]].count += data[0][1]
        else:
            init_node.child[data[0][0]] = fpTreeNode(data[0][0],data[0][1], init_node)
        if len(data) >1:
            conditional_fptree(name,init_node.child[data[0][0]],data[1::])

"""Hàm tìm tập phổ biến"""
def create_leaf_cond_base(header_item_dict, minsup):
    final_cond_base = []#bảng chưa tập đường đi từ lá đến gốc
    final_cond_baseAll = []#bảng chứa tập phổ biến

   # print("header_item_dict.items() ", header_item_dict.items())
    print(" Duyệt các mục trong bảng header")
    for key,value in header_item_dict.items():
        final_cond_base_key = []
        condition_base = []
        leaf_item_count = OrderedDict()#{} 
        print(" ")
        print("duyet key = ", key)
        print(" ")
        # duyệt từ nút lá đến cho đến gốc 
        while value != None:
            path = []
            leaf_node = value
            leaf_count = value.count
       
            while leaf_node.parent != None:
                leaf_details = [leaf_node.name, leaf_count]
                path.append(leaf_details)
                leaf_node = leaf_node.parent # chuyển nút hiện tại sang nút cha để duyệt tiếp
            
            condition_base.insert(0,path)
            # Chuyển đến liên kết tiếp theo
            value = value.link
        # tạo bảng tần suất cho đường dẫn (từ lá đến gốc) của nút đang duyêt
        print("condition_base ", condition_base)
        for row in condition_base:
            for col in row:
                if col[0] not in leaf_item_count:
                    leaf_item_count[col[0]] = col[1]
                else:
                    leaf_item_count[col[0]] += col[1]
        # loại bỏ những mục có tần suất < minsup
        leaf_item_count = {k:v for k,v in leaf_item_count.items() \
                          if v >= minsup}
        print("leaf_item_count ", leaf_item_count)
       
        for row in condition_base:
            temp = []
            temp_tree = []
          #  print("row ", row)
            for col in row:
             #   print("col ", col)
                if col[0] in leaf_item_count:
                    temp.append(col[0]) # tên mục
                    temp_tree.append(col) # tên mục và tần suất
            #print("temp ", temp)
            #print("temp_tree ", temp_tree)
            dataout = []
            createBase(temp, dataout)
            #print("dataout ", dataout)
            if(len(dataout) >0):
                for x in dataout:
                    final_cond_baseAll.append(x) 
            final_cond_base.append(temp)
            final_cond_base_key.append(temp_tree) 
            
        #print("final_cond_base ", final_cond_base)
        #print("final_cond_base_key ", final_cond_base_key)
       
    ## loại bỏ những trùng lặp có trong tập phổ biến
    unique_cond_base_set = set(map(tuple,final_cond_baseAll))
    print("unique_cond_base_set: ",unique_cond_base_set)
    unique_cond_base_list =list(unique_cond_base_set)
    print("unique_cond_base_list: ",unique_cond_base_list)
    #sắp xếp theo bảng chữ cái
    unique_cond_base_list.sort(key=lambda unique_cond_base_list:unique_cond_base_list[0])
    #print("unique_cond_base_list sort: ",unique_cond_base_list)
    #print("unique_cond_base_list: ",unique_cond_base_list)
    unique_cond_base = map(list,unique_cond_base_list)
    export_to_file(unique_cond_base) # xuất ra file csv

def createBase(data, dataout):
    print(data)
    n = range(0, len(data))
    #out = []
    prev = []
    check = False
    for i in n:
        g = i
        n2 = range(g + 1, len(data))
        prev.append(data[g])
        if(check == False):
            dataout.append([data[g]])
            check = True
       # print("prev ",prev)
        for j in n2:
            temp = []
            for r in prev:
                temp.append(r)
            temp.append(data[j])
          #  print("temp ",temp)
            dataout.append(temp)
    #print(dataout) 
def taoMucPhoBien1(name, data, final_cond_base,final_cond_base_key,dataout):
    temp_tree = []
    datanew = []
    
    print("data: ",data)
    if(len(data)>0):
        for col in data:
            temp = []
            print("name: ", name)
            print("col data: ", col)
            temp.append(name)
            temp.append(col)
            print("temp: ",temp)
            dataout.append(temp)
        
        print("datanew ",dataout)
        if (len(data) >1):
            print("data[1::] ",data[1::])
            print("datanew[0] ",datanew[0])
            taoMucPhoBien(datanew[0],data[1::], final_cond_base,final_cond_base_key,dataout)

"""Main"""
# đầu vào


support = 3
file_name = "BRCA.csv"
file_delimiter = ','
output_file_name = "output.csv"

dataset, count_items = fp_tree_preprocess(file_name, support)
print("dataset: ",dataset)
print("")
print("count_items: ",count_items)
fptree_root, header_table = fp_tree_reorder(dataset, count_items)
print("")
print("")
fptree_root.display_tree_list ()
print("")
create_leaf_cond_base(header_table,support)
print("Tập phổ biến được ghi ra trong file output.csv")
