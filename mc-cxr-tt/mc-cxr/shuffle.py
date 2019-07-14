from random import shuffle

f = open('train-data.csv', 'r')
l = f.readlines()
f.close()

head = l[0]
data = l[1:]

shuffle(data)

f = open('train-data-shuffled.csv', 'w')
f.write(head)
for row in data:
    f.write(row)
f.close()
