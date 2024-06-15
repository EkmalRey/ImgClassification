import os
import cv2
import shutil
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB3, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

class DataPreparation:
    def __init__(self, dataset_folder, img_size=(224, 224, 3), batch_size=32):
        self.dataset_folder = dataset_folder
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = []
        self.class_count = 0
        self.data_setup = {
            'Resized': False,
            'ErrorChecked': False,
            'CopyToLocalFolder': False,
            'Augmented': False,
            'Upsampled': False,
        }

    def load_dataset(self, resize=False, check_error=False, copy_to_local=False):
        self.data_setup['Resized'] = resize
        self.data_setup['ErrorChecked'] = check_error
        self.data_setup['CopyToLocalFolder'] = copy_to_local
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
                if resize:
                  img = cv2.resize(img, self.img_size[:2])
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
        print(f"Classes: {', '.join(self.classes)}")
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
        self.data_setup['Upsampled'] = upsample
        train_df, temp_df = train_test_split(data, train_size=train_size, stratify=data['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'])

        train_counts = train_df['label'].value_counts().sort_index()
        val_counts = val_df['label'].value_counts().sort_index()
        test_counts = test_df['label'].value_counts().sort_index()
        train_additional = pd.Series(0, index=train_counts.index)
        val_additional = pd.Series(0, index=val_counts.index)

        if upsample:
            train_df = self.upsample_minority(train_df)
            val_df = self.upsample_minority(val_df)
            train_additional = train_df['label'].value_counts().sort_index() - train_counts
            val_additional = val_df['label'].value_counts().sort_index() - val_counts
            pass

        max_rows = max(len(train_counts), len(val_counts), len(test_counts))
        print(f"{'':<3} {'Label':<15} {'Train Count':<20} {'Val Count':<20} {'Test Count':<15}")
        for i in range(max_rows):
            label = train_counts.index[i] if i < len(train_counts) else ""
            train_count = train_counts[label] if label in train_counts else ""
            train_add = train_additional[label] if label in train_additional else ""
            val_count = val_counts[label] if label in val_counts else ""
            val_add = val_additional[label] if label in val_additional else ""
            test_count = test_counts[label] if label in test_counts else ""
            print(f"{i:<3} {label:<15} {train_count:<{5}} +{train_add:<13} {val_count:<{5}} +{val_add:<13} {test_count:<{15}}")
        print(f"{'':<3} {'Total':<15} {len(train_df):<20} {len(val_df):<20} {len(test_df):<15}")
        return train_df, val_df, test_df, train_additional

    def upsample_minority(self, dataframe):
        upsampled_data = dataframe.copy()
        max_class_count = upsampled_data['label'].value_counts().max()
        for label in self.classes:
            label_data = upsampled_data[upsampled_data['label'] == label]
            label_count = len(label_data)
            if label_count < max_class_count:
                n = max_class_count - label_count
                additional_samples = label_data.sample(n=n, replace=True)
                upsampled_data = pd.concat([upsampled_data, additional_samples])
        upsampled_data = upsampled_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
        return upsampled_data

    def image_generator(self, train_df, val_df, test_df, aug_train=False, class_mode='categorical'):
        self.data_setup['Augmented'] = aug_train
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
                                                            batch_size=self.batch_size)

        val_generator = valtest_datagen.flow_from_dataframe(val_df,
                                                            x_col='image_path',
                                                            y_col='label',
                                                            target_size=self.img_size[:2],
                                                            class_mode=class_mode,
                                                            color_mode='rgb',
                                                            shuffle=False,
                                                            batch_size=self.batch_size)
        length = len(test_df)
        test_batch_size = sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]
        test_generator = valtest_datagen.flow_from_dataframe(test_df,
                                                             x_col='image_path',
                                                             y_col='label',
                                                             target_size=self.img_size[:2],
                                                             class_mode=class_mode,
                                                             color_mode='rgb',
                                                             shuffle=False,
                                                             batch_size=test_batch_size)
        return train_generator, val_generator, test_generator

    def show_sample(self, generator):
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
    def __init__(self, model_folder, img_size=(224, 224, 3), classes=[], data_setup=[]):
        self.model_folder = model_folder
        if os.path.exists(os.path.join(self.model_folder, '_Model_List.csv')):
            self.model_df = pd.read_csv(os.path.join(self.model_folder, '_Model_List.csv'))
        else:
            self.model_df = pd.DataFrame(columns=['ModelId', 'Base', 'Learning Rate', 'Epochs', 'Layers', 'Data Processing', 'Class_Count', 'F1-Score'])
            self.model_df.to_csv(os.path.join(self.model_folder, '_Model_List.csv'), index=False)
        self.img_size = img_size
        self.classes = classes
        self.class_count = len(self.classes)
        self.data_setup = data_setup
        self.model_setup = {
            'ModelId': len(self.model_df),
            'Base': None,
            'Learning Rate': None,
            'Epochs':0,
            'Layers': [],
            'Data Processing': [key for key, value in self.data_setup.items() if value],
            'Class_Count': self.class_count,
            'F1-Score': None
        }

    def create_basemodel_keras(self, base, lr=0.001, weights="imagenet",  trainable=False, pooling=None):
        if base == "EffNet":
            base_model = EfficientNetB3(include_top=False,
                                        weights=weights,
                                        input_shape=self.img_size,
                                        pooling=pooling
                                        )
        elif base == "VGG16":
            base_model = VGG16(include_top=False,
                                weights=weights,
                                input_shape=self.img_size,
                                pooling=pooling
                               )
        else:
            raise ValueError("Available base models: EffNet, VGG16")

        self.model_setup['Base'] = base
        self.model_setup['Layers'].append('FE')
        self.learning_rate = lr
        self.model_setup['Learning Rate'] = lr

        base_model.trainable = trainable
        print(f"Created a Keras {base} base model for feature extraction!")

        return base_model

    def add_fully_connected_layers(self, base_model, layer_units, activation='relu'):
        x = base_model.output
        for units in layer_units:
            x = Dense(units, activation=activation)(x)
            self.model_setup['Layers'].append(f'FC{units}')
        new_model = Model(inputs=base_model.input, outputs=x)
        print(f"Added {len(layer_units)} fully connected layers to the base model with {layer_units} neurons and {activation} activation on it.")
        return new_model

    def add_convolutional_layers(self, base_model, num_layers, filters, kernel_size, pool_size, activation='relu'):
        x = base_model.output
        for _ in range(num_layers):
            x = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)(x)
            x = MaxPooling2D(pool_size=pool_size)(x)
            self.model_setup['Layers'].append(f"C{filters}{activation}")
        new_model = Model(inputs=base_model.input, outputs=x)
        self.model_desc['Additional_Layers'] += num_layers
        print(f"Added {num_layers} convolutional layers with {filters} filters, {kernel_size} kernel size, {pool_size} pool size, {activation} activation.")
        return new_model

    def add_batch_normalization(self, base_model, axis=-1, momentum=0.99, epsilon=0.001):
        x = base_model.output
        x = BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon)(x)
        self.model_setup['Layers'].append(f"BN")
        new_model = Model(inputs=base_model.input, outputs=x)
        print(f"Added BatchNormalization layer with momentum={momentum} and epsilon={epsilon}.")
        return new_model

    def add_dropout(self, base_model, rate=0.4):
        x = base_model.output
        x = Dropout(rate=rate)(x)
        self.model_setup['Layers'].append(f"D{rate}")
        new_model = Model(inputs=base_model.input, outputs=x)
        print(f"Added Dropout layer with rate={rate}.")
        return new_model

    def add_flatten(self, base_model):
        x = base_model.output
        x = Flatten()(x)
        self.model_setup['Layers'].append(f"F")
        new_model = Model(inputs=base_model.input, outputs=x)
        print(f"Added Flatten layer.")
        return new_model

    def model_compile(self, base_model):
        x = base_model.output
        output = Dense(self.class_count, activation='softmax')(x)
        final_model = Model(inputs=base_model.input, outputs=output)
        final_model.compile(Adamax(learning_rate=self.learning_rate),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        print(f"Finalized model with {self.class_count} output classes.")
        return final_model

    def train(self, base_model, train_gen, val_gen, epochs=10, callbacks=None):
        folder_name = f"{self.model_setup['ModelId']}_{self.model_setup['Base']}_LR{self.model_setup['Learning Rate']}_{'-'.join(map(str, self.model_setup['Layers']))}_{'-'.join(map(str, self.model_setup['Data Processing']))}"
        if callbacks == None:
            callbacks =  [
                ModelCheckpoint(filepath=os.path.join(self.model_folder, folder_name, 'model_best.keras'), monitor='val_accuracy', save_best_only=True, mode='max'),
                ModelCheckpoint(filepath=os.path.join(self.model_folder, folder_name, 'model_last.keras'), save_freq=5)
            ]
        os.makedirs(os.path.join(self.model_folder, folder_name), exist_ok=True)
        history = base_model.fit(x=train_gen,
                                validation_data=val_gen,
                                batch_size=train_gen.batch_size,
                                epochs=epochs,
                                verbose=1,
                                callbacks=callbacks,
                                validation_steps=None,
                                shuffle=False
                                 )
        self.model_setup['Epochs'] += epochs
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(self.model_folder, folder_name, 'training_history.csv'), index=False)
        return history

    def visualize_train(self, history):
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

    def export_model(self, model):
        folder_name = f"{self.model_setup['ModelId']}_{self.model_setup['Base']}_LR{self.model_setup['Learning Rate']}_{'-'.join(map(str, self.model_setup['Layers']))}_{'-'.join(map(str, self.model_setup['Data Processing']))}"
        folder = os.path.join(self.model_folder, folder_name)
        os.makedirs(folder, exist_ok=True)
        model.save(f"{folder}/model_manual.keras")

        with open(f'{folder}/architecture.json', 'w') as json_architecture:
            architecture = json.dumps(model.to_json(), indent=4)
            json_architecture.write(architecture)

        with open(f'{folder}/setup.json', 'w') as json_setup:
            setup = json.dumps(self.model_setup, indent=4)
            json_setup.write(setup)

        model_setup = pd.DataFrame([self.model_setup])
        self.model_df = pd.concat([self.model_df, model_setup], axis=0, ignore_index=False)
        self.model_df = self.model_df[~self.model_df.index.duplicated(keep='last')].reset_index(drop=True)
        self.model_df.to_csv(os.path.join(self.model_folder, '_Model_List.csv'), index=False)

    def import_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}.")
            return None
        try:
            model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}.")
            self.model = model
            setup_path = os.path.join(os.path.dirname(model_path), 'setup.json')
            if os.path.exists(setup_path):
                with open(setup_path, 'r') as f:
                    if os.path.getsize(setup_path) > 0:
                        self.model_setup = json.load(f)
                        if self.model_setup['Data Processing'] != [key for key, value in self.data_setup.items() if value]:
                            print("WARNING! The Current Data Processing is not the same as the one in loaded model!")

            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def evaluate(self, model, test_gen):
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

        folder_name = f"{self.model_setup['ModelId']}_{self.model_setup['Base']}_LR{self.model_setup['Learning Rate']}_{'-'.join(map(str, self.model_setup['Layers']))}_{'-'.join(map(str, self.model_setup['Data Processing']))}"
        folder = os.path.join(self.model_folder, folder_name)
        os.makedirs(folder, exist_ok=True)
        prediction_df.to_csv(f"{folder}/test_gen.csv", index=False)

    def classify(self, model, images):
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
        folder_name = f"{self.model_setup['ModelId']}_{self.model_setup['Base']}_LR{self.model_setup['Learning Rate']}_{'-'.join(map(str, self.model_setup['Layers']))}_{'-'.join(map(str, self.model_setup['Data Processing']))}"
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
        if isinstance(img, str):  # If image path
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):  # If image vector
            img = np.array(img)
        else:
            raise ValueError("Unsupported input format. Please provide either a list of image paths or a list of image vectors.")

        img = cv2.resize(img, self.img_size[:2])
        img = np.expand_dims(img, axis=0)
        return img