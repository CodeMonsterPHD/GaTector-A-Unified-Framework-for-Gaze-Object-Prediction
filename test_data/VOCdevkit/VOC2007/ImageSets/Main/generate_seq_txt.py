f=open('test.txt','w+')
for i in range(125):
    temp=str(i)
    temp=temp.zfill(4)
    f.write(temp+'\n')