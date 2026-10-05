{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyNTFtWxwgb+RPjnusrijB1j",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/vinodiniilam/Textile_Fabric_Classification/blob/main/Textile_Fabric_Classification.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cW-NFb5cmWCa"
      },
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "\n",
        "print(os.listdir('/content/drive/MyDrive/DATASET'))"
      ],
      "metadata": {
        "id": "p-TDq0GP1lE0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import zipfile\n",
        "import os\n",
        "\n",
        "zip_path = '/content/drive/MyDrive/DATASET/archive.zip'\n",
        "extract_path = '/content/dataset'\n",
        "\n",
        "with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
        "    zip_ref.extractall(extract_path)\n",
        "\n",
        "print(\"Dataset extracted successfully!\")\n",
        "print(os.listdir(extract_path))"
      ],
      "metadata": {
        "id": "eO9qHWHt2j6-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "\n",
        "print(\"Train classes:\")\n",
        "print(os.listdir('/content/dataset/train'))\n",
        "\n",
        "print(\"\\nValidation classes:\")\n",
        "print(os.listdir('/content/dataset/valid'))\n",
        "\n",
        "print(\"\\nTest classes:\")\n",
        "print(os.listdir('/content/dataset/test'))"
      ],
      "metadata": {
        "id": "ARnnO6fx3F6f"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "\n",
        "print(\"TensorFlow version:\", tf.__version__)\n",
        "print(\"GPU:\", tf.config.list_physical_devices('GPU'))"
      ],
      "metadata": {
        "id": "uR46NCfH3gCd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "\n",
        "IMG_SIZE = (128, 128)\n",
        "BATCH_SIZE = 32\n",
        "\n",
        "train_data = tf.keras.utils.image_dataset_from_directory(\n",
        "    '/content/dataset/train',\n",
        "    image_size=IMG_SIZE,\n",
        "    batch_size=BATCH_SIZE\n",
        ")\n",
        "\n",
        "valid_data = tf.keras.utils.image_dataset_from_directory(\n",
        "    '/content/dataset/valid',\n",
        "    image_size=IMG_SIZE,\n",
        "    batch_size=BATCH_SIZE\n",
        ")\n",
        "\n",
        "test_data = tf.keras.utils.image_dataset_from_directory(\n",
        "    '/content/dataset/test',\n",
        "    image_size=IMG_SIZE,\n",
        "    batch_size=BATCH_SIZE\n",
        ")\n",
        "\n",
        "print(\"Dataset loaded successfully!\")"
      ],
      "metadata": {
        "id": "JgAMtyKz6ixf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(\n",
        "    optimizer='adam',\n",
        "    loss='sparse_categorical_crossentropy',\n",
        "    metrics=['accuracy']\n",
        ")\n",
        "\n",
        "print(\"Model compiled successfully!\")"
      ],
      "metadata": {
        "id": "A00UxKwc8EYM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "history = model.fit(\n",
        "    train_data,\n",
        "    validation_data=valid_data,\n",
        "    epochs=20\n",
        ")"
      ],
      "metadata": {
        "id": "9biUMS6t8lOw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_loss, test_accuracy = model.evaluate(test_data)\n",
        "\n",
        "print(\"Test Loss:\", test_loss)\n",
        "print(\"Test Accuracy:\", test_accuracy)"
      ],
      "metadata": {
        "id": "4M-rO3Pa9dIu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "plt.figure(figsize=(12, 5))\n",
        "\n",
        "# Accuracy\n",
        "plt.subplot(1, 2, 1)\n",
        "\n",
        "plt.plot(\n",
        "    history.history['accuracy'],\n",
        "    label='Training Accuracy'\n",
        ")\n",
        "\n",
        "plt.plot(\n",
        "    history.history['val_accuracy'],\n",
        "    label='Validation Accuracy'\n",
        ")\n",
        "\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.title('Training vs Validation Accuracy')\n",
        "plt.legend()\n",
        "\n",
        "\n",
        "# Loss\n",
        "plt.subplot(1, 2, 2)\n",
        "\n",
        "plt.plot(\n",
        "    history.history['loss'],\n",
        "    label='Training Loss'\n",
        ")\n",
        "\n",
        "plt.plot(\n",
        "    history.history['val_loss'],\n",
        "    label='Validation Loss'\n",
        ")\n",
        "\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Loss')\n",
        "plt.title('Training vs Validation Loss')\n",
        "plt.legend()\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "TA35SaFY98Hz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('/content/drive/MyDrive/DATASET/cnn_textile_model.h5')\n",
        "\n",
        "print(\"Model saved successfully!\")"
      ],
      "metadata": {
        "id": "0S8-QeaN-OXt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "class_names = train_data.class_names\n",
        "\n",
        "for images, labels in test_data.take(1):\n",
        "    predictions = model.predict(images)\n",
        "\n",
        "    for i in range(5):\n",
        "        predicted_class = class_names[np.argmax(predictions[i])]\n",
        "        actual_class = class_names[labels[i].numpy()]\n",
        "\n",
        "        print(\"Actual:\", actual_class)\n",
        "        print(\"Predicted:\", predicted_class)\n",
        "        print(\"--------------------\")\n",
        "\n",
        "        plt.imshow(images[i].numpy().astype(\"uint8\"))\n",
        "        plt.title(f\"Actual: {actual_class} | Predicted: {predicted_class}\")\n",
        "        plt.axis(\"off\")\n",
        "        plt.show()"
      ],
      "metadata": {
        "id": "8kHhnY2o-8ai"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "\n",
        "y_true = []\n",
        "y_pred = []\n",
        "\n",
        "for images, labels in test_data:\n",
        "\n",
        "    predictions = model.predict(\n",
        "        images,\n",
        "        verbose=0\n",
        "    )\n",
        "\n",
        "    y_true.extend(\n",
        "        labels.numpy()\n",
        "    )\n",
        "\n",
        "    y_pred.extend(\n",
        "        np.argmax(predictions, axis=1)\n",
        "    )\n",
        "\n",
        "class_names = test_data.class_names"
      ],
      "metadata": {
        "id": "_0VrH4dTdZLl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "from sklearn.metrics import classification_report\n",
        "\n",
        "print(\n",
        "    classification_report(\n",
        "        y_true,\n",
        "        y_pred,\n",
        "        target_names=class_names\n",
        "    )\n",
        ")"
      ],
      "metadata": {
        "id": "CBQO3epXdp1k"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "y_true = []\n",
        "y_pred = []\n",
        "\n",
        "for images, labels in test_data:\n",
        "    predictions = model.predict(images, verbose=0)\n",
        "\n",
        "    y_true.extend(labels.numpy())\n",
        "    y_pred.extend(np.argmax(predictions, axis=1))\n",
        "\n",
        "cm = confusion_matrix(y_true, y_pred)\n",
        "\n",
        "disp = ConfusionMatrixDisplay(\n",
        "    confusion_matrix=cm,\n",
        "    display_labels=class_names\n",
        ")\n",
        "\n",
        "disp.plot()\n",
        "plt.title(\"Confusion Matrix\")\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "yWDKGoiS_mGp"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}