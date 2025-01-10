import os
import cv2
import json
import time
import shutil
import importlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten

class DataPreparation:
    def __init__(self, dataset_folder, img_size=(224, 224, 3), batch_size=32):
        '''
        Menginisialisasi kelas DataPreparation untuk menyiapkan dataset (hanya digunakan jika ingin training)
        Parameter = dataset_folder{path ke folder dataset}, img_size{size dari img cont. `(224,224,3)`}, dan batch_size{cont. 16, 32, 64}.
        '''
        self.dataset_folder = dataset_folder
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = []
        self.class_count = 0
        self.data_setup = {
            'Batch Size': self.batch_size,
            'Img Size': self.img_size,
        }

    def load_dataset(self, check_error=False, copy_to_local=False):
        '''
        Load dataset dari folder yang telah diinisialisasi.
        Parameter = check_error{boolean}, copy_to_local{boolean}.
        Output = returns data{dataframe pandas berisikan image_path dan label dari setiap gambar}.
        '''
        self.data_setup['Error Checked'] = check_error
        self.data_setup['Copied to Local'] = copy_to_local

        image_paths = []
        labels = []

        if copy_to_local:
            local_path = '/content/Local_Dataset'
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            shutil.copytree(self.dataset_folder, local_path)
            self.dataset_folder = local_path

        for label in sorted(os.listdir(self.dataset_folder)):
            label_path = os.path.join(self.dataset_folder, label)
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                if check_error:
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Error loading image: {image_path}")
                        os.remove(image_path)
                        continue
                    else:
                        img = cv2.resize(img, (img.shape[1], img.shape[0]))
                        cv2.imwrite(image_path, img)
                image_paths.append(image_path)
                labels.append(label)

        data = pd.DataFrame({
            'image_path': image_paths,
            'label': labels
        })

        self.classes = sorted(data['label'].unique())
        self.class_count = len(self.classes)

        value_counts = data['label'].value_counts().reset_index()
        value_counts.columns = ['Label', 'Count']
        value_counts = value_counts.sort_values('Label').reset_index(drop=True)
        print(value_counts)
        return data

    def describe_dataset(self, data):
        '''
        Menampilkan deskripsi dataset yang telah diload seperti Jumlah Kelas, Kelas, Total Image, Average Img Size, Average Aspect Ratio.
        Parameter = data{dataframe pandas berisikan image_path dan label dari setiap gambar}.
        '''
        print(f"Classes({self.class_count}): {', '.join(self.classes)}")
        print(f"Total number of images: {len(data)}\t||\tNumber of classes: {self.class_count}")
        sample = data.sample(n=50, replace=True)
        ht = 0
        wt = 0
        count = 0
        for i in range(len(sample)):
            path = sample['image_path'].iloc[i]
            try:
                img = cv2.imread(path)
                h = img.shape[0]
                w = img.shape[1]
                wt += w
                ht += h
                count += 1
            except:
                pass
        have = int(ht / count)
        wave = int(wt / count)
        aspect_ratio = have / wave
        print(f'average image height: {have}\t||\taverage image width: {wave}\t||\taspect ratio h/w: {aspect_ratio}')

    def split_dataset(self, data, train_size=0.8, upsample=False):
        '''
        Membagi dataset menjadi train, validation, dan test set.
        Parameter = data{dataframe pandas berisikan image_path dan label dari setiap gambar}, train_size{float}, upsample{boolean}.
        Output =  returns train_df{dataframe pandas berisikan image_path dan label dari setiap gambar untuk training},
                  returns val_df{dataframe pandas berisikan image_path dan label dari setiap gambar untuk validation},
                  returns test_df{dataframe pandas berisikan image_path dan label dari setiap gambar untuk test
        '''
        self.data_setup['Train Size'] = train_size
        self.data_setup['Upsampling'] = upsample

        train_df, temp_df = train_test_split(data, train_size=train_size, stratify=data['label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

        train_counts = train_df['label'].value_counts().sort_index()
        val_counts = val_df['label'].value_counts().sort_index()
        test_counts = test_df['label'].value_counts().sort_index()
        train_additional = pd.Series(0, index=train_counts.index)

        if upsample:
            train_df = self.upsample_minority(train_df)
            train_additional = train_df['label'].value_counts().sort_index() - train_counts
            pass

        max_rows = max(len(train_counts), len(val_counts), len(test_counts))
        print(f"{'':<3} {'Label':<15} {'Train Count':<20} {'Val Count':<15} {'Test Count':<15}")
        for i in range(max_rows):
            label = train_counts.index[i] if i < len(train_counts) else ""
            train_count = train_counts[label] if label in train_counts else ""
            train_add = train_additional[label] if label in train_additional else ""
            val_count = val_counts[label] if label in val_counts else ""
            test_count = test_counts[label] if label in test_counts else ""
            print(f"{i:<3} {label:<15} {train_count:<{5}} +{train_add:<13} {val_count:<{15}} {test_count:<{15}}")
        print(f"{'':<3} {'Total':<15} {len(train_df):<20} {len(val_df):<15} {len(test_df):<15}")
        return train_df, val_df, test_df, train_additional

    def upsample_minority(self, dataframe):
        '''
        Melakukan upsampling terhadap data minoritas, digunakan dalam fungsi split_dataset.
        Parameter = dataframe{dataframe pandas berisikan image_path dan label dari setiap gambar}.
        Output = returns dataframe{dataframe pandas berisikan image_path dan label dari setiap gambar setelah di upsampling}.
        '''
        upsampled_data = dataframe.copy()
        max_class_count = upsampled_data['label'].value_counts().max()
        for label in self.classes:
            label_data = upsampled_data[upsampled_data['label'] == label]
            label_count = len(label_data)
            if label_count < max_class_count:
                n = max_class_count - label_count
                additional_samples = label_data.sample(n=n, replace=True, random_state=42)
                upsampled_data = pd.concat([upsampled_data, additional_samples])
        upsampled_data = upsampled_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return upsampled_data

    def image_generator(self, train_df, val_df, test_df, aug_train=False, class_mode='categorical'):
        '''
        Membuat generator untuk data training, validation, dan test set agar dapat diproses tensorflow.
        parameter = train_df{dataframe pandas berisikan image_path dan label dari setiap gambar untuk training},
                    val_df{dataframe pandas berisikan image_path dan label dari setiap gambar untuk validation},
                    test_df{dataframe pandas berisikan image_path dan label dari setiap gambar untuk test},
                    aug_train{boolean}, dan class_mode{string}.
        output =  returns train_generator{generator untuk data training},
                  returns val_generator{generator untuk data validation},
                  returns test_generator{generator untuk data test}.
        '''
        self.data_setup['Augmented Train'] = aug_train
        self.data_setup['Class Mode'] = class_mode

        if aug_train:
            train_datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='constant',
                cval=0
            )
        else:
            train_datagen = ImageDataGenerator()
        valtest_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_dataframe(train_df,
                                                            x_col='image_path',
                                                            y_col='label',
                                                            target_size=self.img_size[:2],
                                                            class_mode=class_mode,
                                                            color_mode='rgb',
                                                            shuffle=True,
                                                            batch_size=self.batch_size,
                                                            interpolation='bilinear',
                                                            prefetch=AUTOTUNE)

        val_generator = valtest_datagen.flow_from_dataframe(val_df,
                                                            x_col='image_path',
                                                            y_col='label',
                                                            target_size=self.img_size[:2],
                                                            class_mode=class_mode,
                                                            color_mode='rgb',
                                                            shuffle=False,
                                                            batch_size=self.batch_size,
                                                            interpolation='bilinear',
                                                            prefetch=AUTOTUNE)
        length = len(test_df)
        test_batch_size = sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]
        test_generator = valtest_datagen.flow_from_dataframe(test_df,
                                                             x_col='image_path',
                                                             y_col='label',
                                                             target_size=self.img_size[:2],
                                                             class_mode=class_mode,
                                                             color_mode='rgb',
                                                             shuffle=False,
                                                             batch_size=test_batch_size,
                                                             interpolation='bilinear',
                                                             prefetch=AUTOTUNE)
        return train_generator, val_generator, test_generator

    def show_sample(self, generator):
        '''
        Menampilkan 9 gambar acak dari sebuah generator untuk memastikan generator bekerja dengan baik.
        Parameter = generator{generator untuk data training}.
        '''
        t_dict = generator.class_indices
        classes = list(t_dict.keys())
        images, labels = next(generator)
        plt.figure(figsize=(10, 10))
        length = len(labels)
        if length < 9:
            r = length
        else:
            r = 9
        for i in range(r):
            plt.subplot(3, 3, i + 1)
            image = images[i] / 255.0
            plt.imshow(image)
            index = np.argmax(labels[i])
            class_name = classes[index]
            plt.title(class_name, color='blue', fontsize=18)
            plt.axis('off')
        plt.show()

class CNNModel:
    def __init__(self, model_folder, img_size=(224, 224, 3), classes=[], data_setup={}):
        '''
        Menginisialisasi kelas untuk manajemen model.
        Parameter = model_folder{path ke root folder model}, img_size{size dari img cont. `(224,224,3)`}, classes{list of string}, data_setup{string dari data processing}.
        '''
        self.model_folder = model_folder
        self.img_size = img_size
        self.classes = classes
        self.class_count = len(self.classes)
        self.dtype_map={
            'Name': str,
            'Created': str,
            'Total Runtime': float,
            'Base': str,
            'Trainable': bool,
            'Learning Rate': float,
            'Layers': str,
            'Epochs': int,
            'F1-Score': float,
            'train_acc': float,
            'train_loss': float,
            'val_acc': float,
            'val_loss': float,
            'Batch Size': int,
            'Train Size': float,
            'Img Size': object,
            'Error Checked': bool,
            'Copied to Local': bool,
            'Upsampling': bool,
            'Augmented Train': bool,
            'Class Mode': str,
            'Classes': object,
        }
        if os.path.exists(f"{self.model_folder}/_Model_List.xlsx"):
            self.model_df = pd.read_excel(f"{self.model_folder}/_Model_List.xlsx", dtype=self.dtype_map)
        else:
            self.model_df = pd.DataFrame(columns=[key for key in self.dtype_map.keys()])
            self.model_df.to_excel(os.path.join(self.model_folder, '_Model_List.xlsx'), index=False)

        self.model_setup = {
            'Name': f"Unknown-{len(self.model_df)}",
            'Created': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() + 25200)),
            'Total Runtime': 0.0,
            'Base': '',
            "Trainable": False,
            'Learning Rate': 0,
            'Layers': '',
            'Epochs':0,
            'F1-Score': 0,
            'train_acc': 0,
            'train_loss': 0,
            'val_acc': 0,
            'val_loss': 0,
            'Batch Size': 0,
            'Train Size': 0,
            'Img Size': self.img_size,
            'Error Checked': False,
            'Copied to Local': False,
            'Upsampling': False,
            'Augmented Train': False,
            'Class Mode': 'categorical',
            'Classes': self.classes
        }
        self.model_setup.update(data_setup)

    def create_basemodel_keras(self, base, lr=0.001,  trainable=False, pooling=None):
        '''
        Membuat basemodel dari keras application.
        Parameter = base{string tipe model}, lr{float}, weights{string}, trainable{boolean untuk menentukan apakah basemodel trainable}, pooling{string `'max','average'`}.
        Output = returns tensorflow model.
        '''
        self.learning_rate = lr
        self.model_setup['Learning Rate'] = lr
        self.model_setup['Trainable'] = trainable
        self.model_setup['Base'] = base
        self.model_setup['Name'] = self.model_setup['Name'].replace("Unknown", base)
        if base not in ["Xception", "VGG16", "VGG19", "ResNet50", "ResNet50V2", "ResNet101", "ResNet101V2", "ResNet152", "ResNet152V2", "InceptionV3", "InceptionResNetV2", "MobileNet", "MobileNetV2", "DenseNet121", "DenseNet169", "DenseNet201", "NASNetMobile", "NASNetLarge", "EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3", "EfficientNetB4", "EfficientNetB5", "EfficientNetB6", "EfficientNetB7", "EfficientNetV2B0", "EfficientNetV2B1", "EfficientNetV2B2", "EfficientNetV2B3", "EfficientNetV2S", "EfficientNetV2M", "EfficientNetV2L"]:
            raise ValueError(f"Available base models: Xception, VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201, NASNetMobile, NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L")
        if pooling not in [None, 'max', 'average']:
            raise ValueError(f"Available pooling types: None, 'max', 'average'")
        module = importlib.import_module(f"tensorflow.keras.applications")
        BaseClass = getattr(module, base)
        base_model = BaseClass(include_top=False,
                                weights="imagenet",
                                input_shape=self.img_size,
                                pooling=pooling
                                )
        self.model_setup['Layers'] += "FE"
        if pooling: self.model_setup['Layers'] += f"-{pooling.capitalize()}Pool"

        if trainable:
            base_model.trainable = trainable
        print(f"Created a Keras {base} base model for feature extraction!")

        return base_model

    def add_fully_connected_layers(self, base_model, layer_units, activation='relu'):
        '''
        Menambahkan fully connected layers pada base model.
        Parameter = base_model{tensorflow model}, layer_units{list of int}, activation{string}.
        Output = returns tensorflow model yang sudah ditambahkan fully connected layer(s).
        '''
        x = base_model.output
        for units in layer_units:
            x = Dense(units, activation=activation)(x)
            self.model_setup['Layers'] = '-'.join([self.model_setup['Layers'], f"FC{units}"])
        new_model = Model(inputs=base_model.input, outputs=x)
        print(f"Added {len(layer_units)} fully connected layers to the base model with {layer_units} neurons and {activation} activation on it.")
        return new_model

    def add_convolutional_layers(self, base_model, num_layers, filters, kernel_size, pool_size, activation='relu'):
        '''
        Menambahkan convolutional layers pada base model.
        Parameter = base_model{tensorflow model}, num_layers{int}, filters{int}, kernel_size{int}, pool_size{int}, activation{string}.
        Output = returns tensorflow model yang sudah ditambahkan convolutional layer(s).
        '''
        x = base_model.output
        for _ in range(num_layers):
            x = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)(x)
            x = MaxPooling2D(pool_size=pool_size)(x)
            self.model_setup['Layers'] = '-'.join([self.model_setup['Layers'], f"C{filters}{activation}"])
        new_model = Model(inputs=base_model.input, outputs=x)
        self.model_setup['Additional_Layers'] += num_layers
        print(f"Added {num_layers} convolutional layers with {filters} filters, {kernel_size} kernel size, {pool_size} pool size, {activation} activation.")
        return new_model

    def add_batch_normalization(self, base_model, axis=-1, momentum=0.99, epsilon=0.001):
        '''
        Menambahkan batch normalization layer pada base model.
        Parameter = base_model{tensorflow model}, axis{int}, momentum{float}, epsilon{float}.
        Output = returns tensorflow model yang sudah ditambahkan batch normalization layer.
        '''
        x = base_model.output
        x = BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon)(x)
        self.model_setup['Layers'] = '-'.join([self.model_setup['Layers'], f"BN"])
        new_model = Model(inputs=base_model.input, outputs=x)
        print(f"Added BatchNormalization layer with momentum={momentum} and epsilon={epsilon}.")
        return new_model

    def add_dropout(self, base_model, rate=0.4):
        '''
        Menambahkan dropout layer pada base model.
        Parameter = base_model{tensorflow model}, rate{float}.
        Output = returns tensorflow model yang sudah ditambahkan dropout layer.
        '''
        x = base_model.output
        x = Dropout(rate=rate)(x)
        self.model_setup['Layers'] = '-'.join([self.model_setup['Layers'], f"D{rate}"])
        new_model = Model(inputs=base_model.input, outputs=x)
        print(f"Added Dropout layer with rate={rate}.")
        return new_model

    def add_flatten(self, base_model):
        '''
        Menambahkan flatten layer pada base model.
        Parameter = base_model{tensorflow model}.
        Output = returns tensorflow model yang sudah ditambahkan flatten layer.
        '''
        x = base_model.output
        x = Flatten()(x)
        self.model_setup['Layers'] = '-'.join([self.model_setup['Layers'], f"F"])
        new_model = Model(inputs=base_model.input, outputs=x)
        print(f"Added Flatten layer.")
        return new_model

    def model_compile(self, base_model, force=False):
        '''
        Compiling model yang sudah dibuat.
        Parameter = base_model{tensorflow model}, force{boolean}.
        Output = returns tensorflow model yang sudah dicompile dan siap untuk digunakan/ditraining.
        '''
        folder_exists = self._model_exists(self.model_df, self.model_setup)
        if folder_exists and not force:
            print(f"Model with setup {folder_exists} already exists! Loading that model instead...")
            model = self.import_model(os.path.join(self.model_folder, folder_exists, 'model.keras'))
            return model
        x = base_model.output
        output = Dense(self.class_count, activation='softmax')(x)
        final_model = Model(inputs=base_model.input, outputs=output)
        final_model.compile(Adamax(learning_rate=self.learning_rate),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        print(f"Finalized model with {self.class_count} output classes.")
        return final_model

    def create_model(self, architecture):
        base_model = None
        for layer, params in architecture:
            if layer.lower() == 'fe':
                base_model = self.create_basemodel_keras(**params)
            elif layer.lower() == 'fc':
                base_model = self.add_fully_connected_layers(base_model, params)
            elif layer.lower() == 'c':
                base_model = self.add_convolutional_layers(base_model, **params)
            elif layer.lower() == 'bn':
                base_model = self.add_batch_normalization(base_model, **params)
            elif layer.lower() == 'd':
                base_model = self.add_dropout(base_model)
            elif layer.lower() == 'f':
                base_model = self.add_flatten(base_model)
            else:
                raise ValueError(f"Invalid layer type: {layer}")

        compiled_model = self.model_compile(base_model)
        return compiled_model

    def train(self, base_model, train_gen, val_gen, epochs=10, callbacks=None, arch_only=False):
        '''
        Melatih model yang sudah dicompile.
        Parameter = base_model{tensorflow model}, train_gen{generator untuk data training}, val_gen{generator untuk data validation}, epochs{int}, callbacks{list of callbacks}.
        Output = returns history dari training model yang bisa digunakan untuk visualisasi.
        '''
        folder_name = f"{self.model_setup['Name']}"
        os.makedirs(os.path.join(self.model_folder, folder_name), exist_ok=True)
        if callbacks == None:
            callbacks =  [
                CustomCheckpoint(filepath=os.path.join(self.model_folder, folder_name, 'model'), setup_path=os.path.join(self.model_folder, folder_name, 'setup.json'), setup=self.model_setup, save_freq=1, arch_only=arch_only)
            ]
        history = base_model.fit(x=train_gen,
                                validation_data=val_gen,
                                batch_size=train_gen.batch_size,
                                epochs=epochs,
                                verbose=1,
                                callbacks=callbacks,
                                shuffle=False
                                )
        history_df = pd.DataFrame(history.history)
        if os.path.exists(os.path.join(self.model_folder, folder_name, 'training_history.csv')):
            history_df_existing = pd.read_csv(os.path.join(self.model_folder, folder_name, 'training_history.csv'))
            history_df = pd.concat([history_df_existing, history_df], ignore_index=True)
        history_df.to_csv(os.path.join(self.model_folder, folder_name, 'training_history.csv'), index=False)
        setup_path = os.path.join(self.model_folder, folder_name, 'setup.json')
        with open(setup_path, 'r') as f:
            self.model_setup = json.load(f)
        return history

    def visualize_train(self, history):
        '''
        Mengvisualisasikan history training model.
        Parameter = history{history dari training model}.
        '''
        tacc = history.history['accuracy']
        tloss = history.history['loss']
        vacc = history.history['val_accuracy']
        vloss = history.history['val_loss']

        Epoch_count = len(tacc)
        Epochs = range(1, Epoch_count+1)

        index_loss = np.argmin(vloss)
        val_lowest = vloss[index_loss]
        index_acc = np.argmax(vacc)
        acc_highest = vacc[index_acc]

        plt.style.use('fivethirtyeight')
        sc_label='best epoch= '+ str(index_loss+1)
        vc_label='best epoch= '+ str(index_acc+1)
        fig, axs=plt.subplots(nrows=1, ncols=2, figsize=(25,10))

        axs[0].plot(Epochs,tloss, 'r', label='Training loss')
        axs[0].plot(Epochs,vloss,'g',label='Validation loss' )
        axs[0].scatter(index_loss+1 ,val_lowest, s=150, c= 'blue', label=sc_label)
        axs[0].scatter(Epochs, tloss, s=100, c='red')
        axs[0].set_title('Training and Validation Loss')
        axs[0].set_xlabel('Epochs', fontsize=18)
        axs[0].set_ylabel('Loss', fontsize=18)
        axs[0].legend()
        axs[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
        axs[1].scatter(Epochs, tacc, s=100, c='red')
        axs[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
        axs[1].scatter(index_acc+1 ,acc_highest, s=150, c= 'blue', label=vc_label)
        axs[1].set_title('Training and Validation Accuracy')
        axs[1].set_xlabel('Epochs', fontsize=18)
        axs[1].set_ylabel('Accuracy', fontsize=18)
        axs[1].legend()
        plt.tight_layout
        plt.show()

    def export_model(self, model, arch_only=False):
        '''
        Export model ke file.
        Parameter = model{tensorflow model}.

        '''
        folder_name = f"{self.model_setup['Name']}"
        folder = os.path.join(self.model_folder, folder_name)
        os.makedirs(folder, exist_ok=True)
        if arch_only:
            with open(f"{folder}/model_architecture.json", 'w') as json_model:
                js_model = model.to_json(indent=4)
                json_model.write(js_model)
        else:
            model.save(f"{folder}/model.keras")

        with open(f'{folder}/setup.json', 'w') as json_setup:
            setup = json.dumps(self.model_setup, indent=4)
            json_setup.write(setup)

        model_setup = pd.DataFrame([self.model_setup])
        model_name = model_setup.loc[0, 'Name']
        model_index = int(model_name.split('-')[-1])
        self.model_df.loc[model_index, :] = model_setup.iloc[0, :]
        self.model_df.to_excel(os.path.join(self.model_folder, '_Model_List.xlsx'), index=False)

    def import_model(self, model_path, from_arch=False):
        '''
        Import model dari file.
        Parameter = model_path{path ke file model}.
        Output = returns tensorflow model.
        '''
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}.")
            raise FileNotFoundError
        try:
            model = load_model(model_path)
            self.model = model
            setup_path = os.path.join(os.path.dirname(model_path), 'setup.json')
            if os.path.exists(setup_path):
                with open(setup_path, 'r') as f:
                    if os.path.getsize(setup_path) > 0:
                        self.model_setup = json.load(f)
                        self.learning_rate = self.model_setup['Learning Rate']
                        self.img_size = self.model_setup['Img Size']
                        self.classes = self.model_setup['Classes']
                        self.class_count = len(self.classes)
            print(f"Model loaded successfully from {model_path}.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def evaluate(self, model, test_gen):
        '''
        Evaluasi model.
        Parameter = model{tensorflow model}, test_gen{generator untuk data testing}.
        '''
        actual_labels = [self.classes[i] for i in test_gen.labels]
        predictions_label = []
        predictions_confidence = []

        prediction = model.predict(test_gen, verbose=1)
        for pred in prediction:
            top_index = np.argsort(pred)[::-1]
            top_label = [self.classes[j] for j in top_index]
            top_conf = [pred[j] for j in top_index]
            predictions_label.append(top_label)
            predictions_confidence.append(top_conf)

        prediction_df = pd.DataFrame({
            'Filename': test_gen.filenames,
            'Actual': actual_labels,
        })
        for i in range(len(top_index)):
            prediction_df[f'Prediction {i+1} Label'] = [pred[i] for pred in predictions_label]
            prediction_df[f'Prediction {i+1} Confidence'] = [conf[i] for conf in predictions_confidence]

        folder_name = f"{self.model_setup['Name']}"
        folder = os.path.join(self.model_folder, folder_name)
        os.makedirs(folder, exist_ok=True)
        prediction_df.to_csv(f"{folder}/test_gen.csv", index=False)

    def classify(self, model, images):
        '''
        Mengkliasifikasi suatu gambar atau list gambar.
        Parameter = model{tensorflow model}, images{path to image or list of path to images}.
        Output = returns label prediksi.
        '''
        if isinstance(images, list):
            predictions = []
            for img in images:
                img = self._process_image(img)
                prediction = model.predict(img)
                predicted_class_index = np.argmax(prediction)
                predicted_class = self.classes[predicted_class_index]
                predictions.append(predicted_class)
            return predictions
        else:
            img = self._process_image(images)
            prediction = model.predict(img)
            predicted_class_index = np.argmax(prediction)
            predicted_class = self.classes[predicted_class_index]
            return predicted_class

    def performance(self, k=1):
        '''
        Menampilkan performance model.
        Parameter = k{int}.
        '''
        folder_name = f"{self.model_setup['Name']}"
        predictions_df = pd.read_csv(f'{self.model_folder}/{folder_name}/test_gen.csv')
        actual_labels = predictions_df['Actual']
        prediction_labels = predictions_df.iloc[:, 2:]
        if k > self.class_count:
            raise ValueError(f"k={k} is greater than the number of predictions ({self.class_count})")
        y_true = []
        y_pred = []

        for i in range(len(actual_labels)):
            y_true.append(actual_labels[i])
            if actual_labels[i] in prediction_labels.iloc[i, :k].values:
                y_pred.append(actual_labels[i])
            else:
                y_pred.append(prediction_labels.iloc[i, 0])

        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)

        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        if k == 1:
            weighted_f1 = report_dict['weighted avg']['f1-score']
            self.model_setup['F1-Score'] = weighted_f1

        metrics_df = pd.DataFrame(report_dict).transpose()
        folder = os.path.join(self.model_folder, folder_name)
        os.makedirs(folder, exist_ok=True)
        metrics_df.to_csv(f"{folder}/Metrics_Top{k}.csv")

        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        print("Confusion Matrix: ")
        df_cm = pd.DataFrame(cm, index=self.classes, columns=self.classes)
        plt.figure(figsize=(7, 7))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        plt.savefig(f"{folder}/Metrics_Top{k}ConfusionMatrix.png", bbox_inches='tight')
        plt.show()


    def _process_image(self, img):
        '''
        Mengubah gambar menjadi tensor digunakan di classify.
        Parameter = img{gambar}.
        Output = returns tensor gambar.
        '''
        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            img = np.array(img)
        else:
            raise ValueError("Unsupported input format. Please provide either a list of image paths or a list of image vectors.")

        img = cv2.resize(img, self.img_size[:2])
        img = np.expand_dims(img, axis=0)
        return img

    def _model_exists(self, df, setup):
        '''
        Memeriksa apakah model sudah pernah dibuat atau belum.
        Parameter = df{dataframe dari model_list}, setup{dictionary}
        Output = returns folder_name jika model sudah pernah dibuat, None jika belum.
        '''
        setup_to_compare = {key: setup[key] for key in setup if key in ['Base', 'Trainable', 'Learning Rate', 'Layers', 'Batch Size', 'Img Size', 'check_error', 'copy_to_local', 'upsample', 'aug_train', 'train_size', 'class_mode', 'Classes']}
        setup_to_compare = {k: '-'.join(map(str, v)) if isinstance(v, (list, tuple)) else v for k, v in setup_to_compare.items()}
        for index, row in df.iterrows():
            row_to_compare = {key: row[key] for key in row.index if key in ['Base', 'Trainable', 'Learning Rate', 'Layers', 'Batch Size', 'Img Size', 'Train Size', 'Error Checked', 'Copied to Local', 'Upsampling', 'Augmented Train', 'Class Mode', 'Classes']}
            row_to_compare = {k: '-'.join(v[1:-1].replace("'", "").split(", ")) if isinstance(v, str) and v.startswith("[") and v.endswith("]") else '-'.join(v) if isinstance(v, list) else v for k, v in row_to_compare.items()}
            compare = zip(setup_to_compare.items(), row_to_compare.items())
            if setup_to_compare == row_to_compare:
                found = row.to_dict()
                folder_name = f"{found['Name']}"
                return folder_name
        return None

class CustomCheckpoint(Callback):
    def __init__(self, filepath, setup_path, setup, save_freq=5, arch_only=False):
        '''
        Custom callback to save the best model.
        Parameters = filepath{path to save model}, setup_path{path to save setup}, setup{dictionary}, save_freq{int}.
        '''
        super(CustomCheckpoint, self).__init__()
        self.filepath = filepath
        self.setup_path = setup_path
        self.save_freq = save_freq
        self.arch_only = arch_only
        self.model_setup = setup
        self.best_val_accuracy = 0.0
        self.best_weights = None
        self._load_setup()

    def _load_setup(self):
        if os.path.exists(self.setup_path):
            with open(self.setup_path, 'r') as json_setup:
                self.model_setup = json.load(json_setup)
                self.best_val_accuracy = self.model_setup.get('val_acc', 0.0)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Debugging: Print the keys in logs to check available metrics
        self.model_setup['Epochs'] += 1
        self.model_setup['train_acc'] = logs.get('accuracy')
        self.model_setup['train_loss'] = logs.get('loss')
        self.model_setup['val_loss'] = logs.get('val_loss')
        self.model_setup['val_acc'] = logs.get('val_accuracy')
        current_val_accuracy = logs.get('val_accuracy')
        if current_val_accuracy is not None and current_val_accuracy > self.best_val_accuracy:
            self.best_weights = self.model.get_weights()
            self.best_val_accuracy = current_val_accuracy
            if not self.arch_only:
                self.model.save(f'{self.filepath}_best.keras')
        if (epoch + 1) % self.save_freq == 0:
            if not self.arch_only:
                self.model.save(f'{self.filepath}.keras')
        with open(self.setup_path, 'w') as json_setup:
            setuptemp = json.dumps(self.model_setup, indent=4)
            json_setup.write(setuptemp)

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        tr_duration = time.time() - self.start_time
        self.model_setup['Total Runtime'] += tr_duration
        with open(self.setup_path, 'w') as json_setup:
            setuptemp = json.dumps(self.model_setup, indent=4)
            json_setup.write(setuptemp)
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))
        print(f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)')