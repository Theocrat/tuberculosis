from random import shuffle

f = open('test-data.csv', 'r')
l = f.readlines()
f.close()

head = l[0]
data = l[1:]

shuffle(data)

f = open('test-data-shuffled.csv', 'w')
f.write(head)
for row in data:
    f.write(row)
f.close()
