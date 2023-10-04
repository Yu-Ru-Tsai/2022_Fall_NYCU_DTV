import matplotlib.pyplot as plt

def plot_learning_curve(train_loss, vaild_loss, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(train_loss)
    x_1 = range(total_steps)
    x_2 = x_1[::len(train_loss) // len(vaild_loss)]
    plt.figure(figsize=(6, 4))
    plt.plot(x_1, train_loss, c='tab:red', label='train')
    plt.plot(x_2, vaild_loss, c='tab:cyan', label='valid')
    plt.ylim(0.0, 2.)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Curve of {}'.format(title))
    plt.legend()
    plt.savefig("loss.jpg")
    
def plot_accuracy_curve(train_accuracy, vaild_accuracy, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(train_accuracy)
    x_1 = range(total_steps)
    x_2 = x_1[::len(train_accuracy) // len(vaild_accuracy)]
    plt.figure(figsize=(6, 4))
    plt.plot(x_1, train_accuracy, c='tab:red', label='train')
    plt.plot(x_2, vaild_accuracy, c='tab:cyan', label='valid')
    plt.ylim(0.0, 1.)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy Curve of {}'.format(title))
    plt.legend()
    plt.savefig("accuracy.jpg")
