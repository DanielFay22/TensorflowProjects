"""
Converts the binary release of the MNIST Dataset to CSV format.
"""

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()


directory = 'C:/Users/Daniel/Desktop/DataSet/'

convert(directory + "train-images.idx3-ubyte", directory + "train-labels.idx1-ubyte",
        directory + "mnist_train.csv", 60000)
convert(directory + "t10k-images.idx3-ubyte", directory + "t10k-labels.idx1-ubyte",
        directory + "mnist_test.csv", 10000)