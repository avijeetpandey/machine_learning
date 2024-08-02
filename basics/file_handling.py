# reading a file
with open('example.txt','r') as file:
    content = file.read()
    print(content)

# another way to read file
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip()) # removes new line character