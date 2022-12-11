import streamlit as st

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import imutils
import cv2
from tensorflow import keras 
import joblib
import io
from PIL import Image

st.markdown("# KNN ❄️")
st.sidebar.markdown("# KNN ❄️")

html = '''
    <h3>Bài toán:</h3>
    <p /align="justify";/>bài toán Classification với 3 classes: Đỏ, Lam, Lục. Mỗi điểm dữ liệu mới (test data point) sẽ được gán label theo màu của điểm mà nó thuộc về. Trong hình này, có một vài vùng nhỏ xem lẫn vào các vùng lớn hơn khác màu. Ví dụ có một điểm màu Lục ở gần góc 11 giờ nằm giữa hai vùng lớn với nhiều dữ liệu màu Đỏ và Lam. Điểm này rất có thể là nhiễu. Dẫn đến nếu dữ liệu test rơi vào vùng này sẽ có nhiều khả năng cho kết quả không chính xác.</p>
    '''
st.markdown(html, unsafe_allow_html=True)
st.image('images/knn.png')

def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val == key:
            return value

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

app_mode = st.selectbox('Chọn bài',['Bai01','Bai02', 'Bai03', 'Bai3a', 'Bai04', 'Bai08']) 

if app_mode=='Bai01':
    
    st.title("Bài 1") 
    np.random.seed(100)
    N = 150

    centers = [[2, 3], [5, 5], [1, 8]]
    n_classes = len(centers)
    data, labels = make_blobs(n_samples=N, 
                            centers=np.array(centers),
                            random_state=1)

    nhom_0 = []
    nhom_1 = []
    nhom_2 = []

    for i in range(0, N):
        if labels[i] == 0:
            nhom_0.append([data[i,0], data[i,1]])
        elif labels[i] == 1:
            nhom_1.append([data[i,0], data[i,1]])
        else:
            nhom_2.append([data[i,0], data[i,1]])

    nhom_0 = np.array(nhom_0)
    nhom_1 = np.array(nhom_1)
    nhom_2 = np.array(nhom_2)
    
    fig, ax = plt.subplots()
    plt.plot(nhom_0[:,0], nhom_0[:,1], 'og', markersize = 2)
    plt.plot(nhom_1[:,0], nhom_1[:,1], 'or', markersize = 2)
    plt.plot(nhom_2[:,0], nhom_2[:,1], 'ob', markersize = 2)
    plt.legend(['Nhom 0', 'Nhom 1', 'Nhom 2'])
    res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=1)
    
    train_data, test_data, train_labels, test_labels = res 
    # default k = n_neighbors = 5
    #         k = 3
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(train_data, train_labels)
    predicted = knn.predict(test_data)
    sai_so = accuracy_score(test_labels, predicted)
    st.write('sai so:', sai_so)

    my_test = np.array([[2.5, 4.0]])
    ket_qua = knn.predict(my_test)
    st.write('Ket qua nhan dang cua', my_test, ' la nhom:', ket_qua[0])
    st.pyplot(fig)
elif (app_mode == 'Bai02'):
    st.title("Bài 02") 
    # take the MNIST data and construct the training and testing split, using 75% of the
    # data for training and 25% for testing
    mnist = datasets.load_digits()
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
        mnist.target, test_size=0.25, random_state=42)

    # now, let's take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
        test_size=0.1, random_state=84)

    st.write("training data points: ", len(trainLabels))
    st.write("validation data points: ", len(valLabels))
    st.write("testing data points: ", len(testLabels))

    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)
    # evaluate the model and update the accuracies list
    score = model.score(valData, valLabels)
    st.write("accuracy = %.2f%%" % (score * 100))

    # loop over a few random digits
    for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
        # grab the image and classify it
        image = testData[i]
        prediction = model.predict(image.reshape(1, -1))[0]

        # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
        # then resize it to 32 x 32 pixels so we can see it better
        image = image.reshape((8, 8)).astype("uint8")

        image = exposure.rescale_intensity(image, out_range=(0, 255))
        image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

        # show the prediction
        st.image(image, clamp=True)
        st.write("I think that digit is: {}".format(prediction))
elif (app_mode =='Bai03'):

    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 

    # 784 = 28x28
    RESHAPED = 784
    X_train = X_train.reshape(60000, RESHAPED)
    X_test = X_test.reshape(10000, RESHAPED) 

    # now, let's take 10% of the training data and use that for validation
    (trainData, valData, trainLabels, valLabels) = train_test_split(X_train, Y_train,
        test_size=0.1, random_state=84)

    model = KNeighborsClassifier()
    model.fit(trainData, trainLabels)

    # save model, sau này ta sẽ load model để dùng 
    def pickle_model(model):
        f = io.BytesIO()
        joblib.dump(model, f)
        return f 

    # Đánh giá trên tập validation
    predicted = model.predict(valData)
    do_chinh_xac = accuracy_score(valLabels, predicted)
    st.write('Độ chính xác trên tập validation: %.0f%%' % (do_chinh_xac*100))

    # Đánh giá trên tập test
    predicted = model.predict(X_test)
    do_chinh_xac = accuracy_score(Y_test, predicted)
    st.write('Độ chính xác trên tập test: %.0f%%' % (do_chinh_xac*100))

    st.download_button("Download Model", data=pickle_model(model), file_name="knn_mnist.pkl")

elif (app_mode =='Bai3a'):
    st.title('Bài 3a')
    
    mnist = keras.datasets.mnist 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 


    index = np.random.randint(0, 9999, 100)
    sample = np.zeros((100,28,28), np.uint8)
    for i in range(0, 100):
        sample[i] = X_test[index[i]]


    # 784 = 28x28
    RESHAPED = 784
    sample = sample.reshape(100, RESHAPED) 
    knn = joblib.load("pages/KNN1/knn_mnist.pkl")
    predicted = knn.predict(sample)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            print('%2d' % (predicted[k]), end='')
            k = k + 1
        st.write()

    digit = np.zeros((10*28,10*28), np.uint8)
    k = 0
    for x in range(0, 10):
        for y in range(0, 10):
            digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
            k = k + 1

    st.image('pages/KNN1/digit.jpg')
elif (app_mode == 'Bai04'):
    
    from PIL import ImageTk, Image
    st.header("Bài 4")
    mnist = keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    index = None
    knn = joblib.load('pages/KNN1/knn_mnist.pkl')
    btn1 = st.button('Create Digit and Recognition')
    if btn1:
        col1,col2 = st.columns([15,20])
        index = np.random.randint(0, 9999, 100)
        digit = np.zeros((10*28,10*28), np.uint8)
        k = 0
        for x in range(0, 10):
            for y in range(0, 10):
                    digit[x*28:(x+1)*28, y*28:(y+1)*28] = X_test[index[k]]
                    k = k + 1  
        with col1:
            st.latex("IMAGE")
            st.write()
            st.write()
            cv2.imwrite('pages/KNN1/digit.jpg', digit)
            image = Image.open('pages/KNN1/digit.jpg')
            st.image(image, caption='IMAGE')
            sample = np.zeros((100,28,28), np.uint8)
            for i in range(0, 100):
                sample[i] = X_test[index[i]]
                
            RESHAPED = 784
            sample = sample.reshape(100, RESHAPED) 
            predicted = knn.predict(sample)
            k = 0
            with col2:
                st.latex("Ket qua nhan dang")
                for x in range(0, 10):
                    ketqua = ''
                    for y in range(0, 10):
                        ketqua = ketqua + '%3d' % (predicted[k])
                        k = k + 1
                    st.write(ketqua )
else:
    st.header("Bài 8")
    bottom_image = st.file_uploader('', type='jpg', key=6)
    if bottom_image is not None:
        image = Image.open(bottom_image)
        st.image(image)
        st.write('Resize: width = 600 - height = 400')
        new_image = image.resize((600, 400))
        st.image(new_image)