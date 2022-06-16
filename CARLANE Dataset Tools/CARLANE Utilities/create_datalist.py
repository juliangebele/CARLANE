import os


class FileSaver(object):
    """
    Create .txt file for images without labels
    """
    def __init__(self):
        self.file = open("real_train.txt", 'a')

    def create_datalist(self, loading_dir):
        for filename in os.listdir(loading_dir):
            if filename.endswith('.jpg'):
                load_file = os.path.join(loading_dir, filename)
                self.file.write(load_file + '\n')
            elif not filename.endswith('.png'):
                self.create_datalist(loading_dir + filename + '/')

    def close_file(self):
        self.file.close()


if __name__ == '__main__':
    fs = FileSaver()
    fs.create_datalist("train/real/")
    fs.close_file()
