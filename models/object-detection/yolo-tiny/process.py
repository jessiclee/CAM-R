import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

print(current_dir)

train_dir = 'C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data/train/images'

# Create and/or truncate train.txt and test.txt
file_train = open('C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data/train/train.txt', 'w')


for pathAndFilename in glob.iglob(os.path.join(train_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write("data/obj" + "/" + title + '.jpg' + "\n")



test_dir = 'C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data/valid/images'
# Create and/or truncate  test.txt
file_test = open('C:/Users/jesle/Desktop/fyp/actual_data/model_evaluation_data/valid/test.txt', 'w')

for pathAndFilename in glob.iglob(os.path.join(test_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_test.write("data/obj" + "/" + title + '.jpg' + "\n")
