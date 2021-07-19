import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load training images and labels
images = pickle.load(open(r"E:\Administrator\Desktop\Lane_line_detection\data\full_CNN_train.p", "rb"))
labels = pickle.load(open(r"E:\Administrator\Desktop\Lane_line_detection\data\full_CNN_labels.p", "rb"))

images = np.array(images)
labels = np.array(labels) / 255.

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=42, shuffle=True)


# print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


class Model:
    def __init__(self):
        self.model = None
        self.epochs = 10
        self.batch_size = 128
        self.input_shape = X_train.shape[1:]

    def build_model(self):
        self.model = Sequential([
            layers.BatchNormalization(input_shape=self.input_shape),
            # Conv-Conv-Pooling-1
            layers.Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv1'),
            layers.Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv2'),
            layers.MaxPooling2D(pool_size=[2, 2]),

            # Conv-Conv--ConvPooling-2
            layers.Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv3'),
            layers.Dropout(0.2),
            layers.Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv4'),
            layers.Dropout(0.2),
            layers.Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv5'),
            layers.Dropout(0.2),
            layers.MaxPooling2D(pool_size=[2, 2]),

            # Conv-Conv-Pooling-3
            layers.Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv6'),
            layers.Dropout(0.2),
            layers.Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv7'),
            layers.Dropout(0.2),
            layers.MaxPooling2D(pool_size=[2, 2]),

            # Upsample-Deconv-Deconv 1
            layers.UpSampling2D(size=[2, 2]),
            layers.Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv1'),
            layers.Dropout(0.2),
            layers.Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv2'),
            layers.Dropout(0.2),

            # Upsample-Deconv-Deconv-Deconv 2
            layers.UpSampling2D(size=[2, 2]),
            layers.Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv3'),
            layers.Dropout(0.2),
            layers.Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv4'),
            layers.Dropout(0.2),
            layers.Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv5'),
            layers.Dropout(0.2),

            # Upsample-Deconv-Deconv 3
            layers.UpSampling2D(size=[2, 2]),
            layers.Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv6'),
            layers.Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Final'),
        ])

        self.model.build(input_shape=[None, 80, 160, 3])
        # print(self.model.summary())

    def train_net(self):
        self.model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])  # metrics=['accuracy']
        """
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                       # validation_split=0.2,
                       validation_data=(X_val, y_val),
                       )_generator
        """
        datagen = ImageDataGenerator(channel_shift_range=0.2)
        datagen.fit(X_train)  # fit方法得到原始图形的统计信息，比如均值、方差等

        # 使用 Python 生成器逐批生成的数据，按批次训练模型。
        self.model.fit(datagen.flow(X_train, y_train, batch_size=self.batch_size,
                                    save_to_dir=r'E:\Administrator\Pictures\data', save_format='jpg'),
                       steps_per_epoch=len(X_train) / self.batch_size,  # 生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=(X_val, y_val)
                       )

        # 训练完成后冻结所有权重
        self.model.trainable = False
        self.model.compile(optimizer='Adam', loss='mean_squared_error')

        # Save model architecture and weights
        self.model.save(r'E:\Administrator\Desktop\Lane_line_detection\model\CNN_model.h5')


if __name__ == "__main__":
    net = Model()
    net.build_model()
    net.train_net()
