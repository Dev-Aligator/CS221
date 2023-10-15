import { bar_plot, mo2ml } from "./assets";
function App() {
  return (
    <div className="devsite-article-body clearfix">
      <div className="devsite-page-title-meta"></div>

      <p></p>

      <div itemScope itemType="http://developers.google.com/ReferenceObject">
        <meta itemProp="name" content="Transfer learning and fine-tuning" />
        <meta itemProp="path" content="Guide & Tutorials" />
        <meta itemProp="property" content="tf.data.experimental.cardinality" />
        <meta itemProp="property" content="tf.expand_dims" />
        <meta itemProp="property" content="tf.keras.Input" />
        <meta itemProp="property" content="tf.keras.Model" />
        <meta itemProp="property" content="tf.keras.Sequential" />
        <meta
          itemProp="property"
          content="tf.keras.applications.mobilenet_v2.MobileNetV2"
        />
        <meta itemProp="property" content="tf.keras.layers.Dense" />
        <meta itemProp="property" content="tf.keras.layers.Dropout" />
        <meta
          itemProp="property"
          content="tf.keras.layers.GlobalAveragePooling2D"
        />
        <meta itemProp="property" content="tf.keras.layers.RandomFlip" />
        <meta itemProp="property" content="tf.keras.layers.RandomRotation" />
        <meta itemProp="property" content="tf.keras.layers.Rescaling" />
        <meta
          itemProp="property"
          content="tf.keras.losses.BinaryCrossentropy"
        />
        <meta itemProp="property" content="tf.keras.metrics.BinaryAccuracy" />
        <meta itemProp="property" content="tf.keras.optimizers.Adam" />
        <meta
          itemProp="property"
          content="tf.keras.optimizers.experimental.RMSprop"
        />
        <meta itemProp="property" content="tf.keras.utils.get_file" />
        <meta
          itemProp="property"
          content="tf.keras.utils.image_dataset_from_directory"
        />
        <meta itemProp="property" content="tf.keras.utils.plot_model" />
        <meta itemProp="property" content="tf.math.sigmoid" />
        <meta itemProp="property" content="tf.where" />
      </div>

      <table className="tfo-notebook-buttons" align="left">
        <td>
          <a
            target="_blank"
            href="https://www.tensorflow.org/tutorials/images/transfer_learning"
          >
            <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
            View on TensorFlow.org
          </a>
        </td>
        <td>
          <a
            target="_blank"
            href="https://colab.research.google.com/drive/16MsoHOBBLYH1jMHSSmfgzox6jQlY8dqh"
          >
            <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
            Run in Google Colab
          </a>
        </td>
        <td>
          <a
            target="_blank"
            href="https://github.com/Dev-Aligator/CS221/blob/master/Solution/VLSP-Restaurant.ipynb"
          >
            <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
            View source on GitHub
          </a>
        </td>
        <td>
          <a href="https://drive.google.com/uc?export=download&id=16MsoHOBBLYH1jMHSSmfgzox6jQlY8dqh">
            <img src="https://www.tensorflow.org/images/download_logo_32px.png" />
            Download notebook
          </a>
        </td>
      </table>

      <p>
        In this tutorial, you will learn how to perform aspect category
        sentiment analysis on{" "}
        <a
          href="https://drive.google.com/file/d/1yjZ0sDD2kAKOZK78MFqWJyaDcCSxnIqN/view?usp=drive_link"
          target="_blank"
        >
          Vietnamese restaurant reviews datasets
        </a>{" "}
        using Logistic Regression with Average Gradient Descent.
      </p>

      <p>
        Aspect category sentiment analysis is the process of classifying text
        data, such as customer reviews or social media comments, into specific
        aspect categories (e.g., "FOOD#QUALITY," "SERVICE#GENERAL") and
        determining the sentiment associated with each aspect (e.g., positive,
        negative, neutral).
      </p>

      <p>
        Instead of employing separate models for each entity, we will implement
        <strong> a single model</strong> capable of classifying sentiments
        across all aspect categories efficiently."
      </p>

      <p>
        In this notebook, Our goal is to build a robust sentiment analysis model
        for Vietnamese text data. We will walk through the following key steps:
      </p>

      <ol>
        <li>Examine and understand the data</li>
        <li>
          Feature Extraction: We use the TF-IDF (Term Frequency-Inverse Document
          Frequency) vectorization technique to convert the preprocessed text
          data into numerical features suitable for machine learning.
        </li>
        <li>
          Model Training
          <ul>
            <li>
              We employ a logistic regression classifier, designed to predict
              sentiments for different aspect categories concurrently
            </li>
            <li>
              This model has the capability to classify text into specific
              aspect categories and assign sentiments such as negative, neutral,
              or positive.
            </li>
          </ul>
        </li>
        <li>Model Evaluation</li>
      </ol>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `import numpy as np
import pandas as pd
import tensorflow as tf
import torch
tf.get_logger().setLevel('ERROR')`,
          }}
        ></code>
      </pre>

      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        2023-10-08 23:27:29.061168: I tensorflow/core/util/port.cc:111] oneDNN
        custom operations are on. You may see slightly different numerical
        results due to floating-point round-off errors from different
        computation orders. To turn them off, set the environment variable
        `TF_ENABLE_ONEDNN_OPTS=0`. 2023-10-08 23:27:29.120509: I
        tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on
        your machine, GPU will not be used. 2023-10-08 23:27:29.358877: E
        tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to
        register cuDNN factory: Attempting to register factory for plugin cuDNN
        when one has already been registered 2023-10-08 23:27:29.358905: E
        tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to
        register cuFFT factory: Attempting to register factory for plugin cuFFT
        when one has already been registered 2023-10-08 23:27:29.360397: E
        tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable
        to register cuBLAS factory: Attempting to register factory for plugin
        cuBLAS when one has already been registered 2023-10-08 23:27:29.501851:
        I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on
        your machine, GPU will not be used. 2023-10-08 23:27:29.504355: I
        tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow
        binary is optimized to use available CPU instructions in
        performance-critical operations. To enable the following instructions:
        AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow
        with the appropriate compiler flags. 2023-10-08 23:27:30.487302: W
        tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning:
        Could not find TensorRT
      </pre>

      <h2 id="data_preparation" data-text="Data preparation">
        Data preparation
      </h2>

      <h3 id="data_download" data-text="Data download">
        Data download
      </h3>

      <p>
        In this tutorial, you will use a dataset containing several thousand
        reviews of restaurant in Vietnamese.{" "}
        <a href="https://drive.google.com/uc?export=download&id=1yjZ0sDD2kAKOZK78MFqWJyaDcCSxnIqN">
          Download
        </a>{" "}
        and extract a zip file containing the csv files, or use the original txt
        files provided by VLSP2018 and follow{" "}
        <a
          href="https://github.com/Dev-Aligator/CS221/blob/master/Solution/csvConverter.py"
          target="_blank"
        >
          this
        </a>{" "}
        to convert it to csv .
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `TRAIN_PATH = 'VLSP2018-SA-train-dev-test/csv/train.csv'
VAL_PATH = 'VLSP2018-SA-train-dev-test/csv/dev.csv'
TEST_PATH = 'VLSP2018-SA-train-dev-test/csv/test.csv'`,
          }}
        ></code>
      </pre>
      <h3 id="data_loading" data-text="Data loading">
        Data loading
      </h3>
      <p>
        To load and process the dataset, we first define a function to read CSV
        files using the{" "}
        <a href="https://pandas.pydata.org/docs/" target="_blank">
          Pandas
        </a>{" "}
        library. The goal is to extract the review text and corresponding labels
        for sentiment analysis. Here's how we do it:
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `def read_csv(url):
  df = pd.read_csv(url)    
  X = df.pop('review')
  y = df.replace({np.nan: 0, 
                'negative': 1, 
                'neutral': 2, 
                'positive': 3}).astype(np.uint8)       
  print('X.shape:', X.shape, 'y.shape:', y.shape)
  return X, y`,
          }}
        ></code>
      </pre>

      <p>
        Now that we have the function to read and process data, let's apply it
        to the training, validation, and test datasets:
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `Xtrain, ytrain = read_csv(TRAIN_PATH)
Xdev,   ydev   = read_csv(VAL_PATH)
Xtest,  ytest  = read_csv(TEST_PATH)`,
          }}
        ></code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        X.shape: (2961,) y.shape: (2961, 12) <br></br>
        X.shape: (1290,) y.shape: (1290, 12) <br></br>
        X.shape: (500,) y.shape: (500, 12)
      </pre>

      <h3 id="data_define_constants" data-text="Define Aspects and Sentiments">
        Define Aspects and Sentiments
      </h3>
      <p>
        Based on the VLSP dataset guidelines, we can determine that our analysis
        will have 12 distinct aspects, each associated with 3 different
        sentiments.
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `aspects = ['FOOD#PRICES',
            'FOOD#QUALITY',
            'FOOD#STYLE&OPTIONS',
            'DRINKS#PRICES',
            'DRINKS#QUALITY',
            'DRINKS#STYLE&OPTIONS',
            'RESTAURANT#PRICES',
            'RESTAURANT#GENERAL',
            'RESTAURANT#MISCELLANEOUS',
            'SERVICE#GENERAL',
            'AMBIENCE#GENERAL',
            'LOCATION#GENERAL']
 
sentiments = ['-', 'o', '+']    # Negative, Neutral, Positive`,
          }}
        ></code>
      </pre>
      <p>
        Next, we'll define a function to convert multi-output data into binary
        multi-label format.
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `def mo2ml(y):
  newcols = [f'{a} {s}' for a in aspects for s in sentiments]

  nrows, ncols = len(y), len(newcols)
  ml = pd.DataFrame(np.zeros((nrows, ncols), dtype='bool'),
                    columns=newcols)
  
  for i, a in enumerate(aspects):
      for j in range(1, 4):
          indices = y[a] == j
          ml.iloc[indices, i * 3 + j - 1] = True

  return ml`,
          }}
        ></code>
      </pre>

      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        FOOD#QUALITY = 2 ==={">"} FOOD#QUALITY - = 0 FOOD#QUALITY o = 0
        FOOD#QUALITY + = 1
      </pre>

      <p>
        <img src={mo2ml} alt="png" />
      </p>

      <p>
        Show a bar plot showing the values for each aspects from the training
        set:
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `fig, ax = plt.subplots(figsize=(12, 6))

# Iterate through each category and plot the values
for category_index, category_name in category_names.items():
  ax.bar(category_name, ytrain_transposed[category_index], label=category_name)
            
# Set labels, legend, and title
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Bar Plot of Categories')
ax.legend()

# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()`,
          }}
        ></code>
      </pre>
      <p>
        <img src={bar_plot} alt="png" />
      </p>

      <h2 id="feature_extraction" data-text="Feature extraction">
        Feature extraction
      </h2>
      <p>
        In this step, we convert our text data into term frequency features
        using CountVectorizer from scikit-learn.
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          from sklearn.feature_extraction.text import CountVectorizer <br></br>
          <br></br>
          vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9)
        </code>
      </pre>

      <aside className="note">
        <strong>Note:</strong>
        <span>
          <strong>`ngram_range`:</strong>
          We define the range of word combinations (unigrams, bigrams, and
          trigrams) to capture more context in the text.
          <strong>`min_df=2`:</strong> specifies that a term must appeart in at
          least 2 documents to be considered as a feature.
          <strong>`max_df=0.9`:</strong> specifies that if a term must appeart
          in more than 90% of the documents in the dataset are excluded.
        </span>
      </aside>

      <p>Transformation</p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          from sklearn.feature_extraction.text import CountVectorizer <br></br>
          <br></br>
          vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9)
        </code>
      </pre>
      <h3 id="data_processing" data-text="Data processing">
        Data processing
      </h3>
      <p>
        Looking at the review data, we find hashtags (#), emojis, different
        letter cases, special characters, and Unicode characters. To prepare the
        data properly, we will define a Text Cleaner class as below:
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `class TextCleanerBase(BaseEstimator, TransformerMixin):
  def __init__(self):
    super().__init__()
        
    # Create preprocessing function
    self.normalize_unicode = partial(unicodedata.normalize, 'NFC')
                    
    def fit(self, X, y=None):
      return self
        
    def transform(self, X):
      if not isinstance(X, pd.Series):
        X = pd.Series(X)
        
      return X.apply(str.lower) 
              .apply(remove_emojis) 
              .apply(self.normalize_unicode)
        
def remove_emojis(text):
  return demoji.replace(text, '')`,
          }}
        ></code>
      </pre>
      <p>
        You have the option to skip this step because its impact on test
        accuracy is minimal. Alternatively, you may explore an advanced method
        to enhance accuracy.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          print(&#39;Number of validation batches: %d&#39; %
          tf.data.experimental.cardinality(validation_dataset))
          print(&#39;Number of test batches: %d&#39; %
          tf.data.experimental.cardinality(test_dataset))
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Number of validation batches&colon; 26 Number of test batches&colon; 6
      </pre>

      <h3
        id="configure_the_dataset_for_performance"
        data-text="Configure the dataset for performance"
      >
        Configure the dataset for performance
      </h3>

      <p>
        Use buffered prefetching to load images from disk without having I/O
        become blocking. To learn more about this method see the
        <a href="https://www.tensorflow.org/guide/data_performance">
          data performance
        </a>
        guide.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          AUTOTUNE = tf.data.AUTOTUNE train_dataset =
          train_dataset.prefetch(buffer_size=AUTOTUNE) validation_dataset =
          validation_dataset.prefetch(buffer_size=AUTOTUNE) test_dataset =
          test_dataset.prefetch(buffer_size=AUTOTUNE)
        </code>
      </pre>
      <h3 id="use_data_augmentation" data-text="Use data augmentation">
        Use data augmentation
      </h3>

      <p>
        When you don&#39;t have a large image dataset, it&#39;s a good practice
        to artificially introduce sample diversity by applying random, yet
        realistic, transformations to the training images, such as rotation and
        horizontal flipping. This helps expose the model to different aspects of
        the training data and reduce
        <a href="https://www.tensorflow.org/tutorials/keras/overfit_and_underfit">
          overfitting
        </a>
        . You can learn more about data augmentation in this
        <a href="https://www.tensorflow.org/tutorials/images/data_augmentation">
          tutorial
        </a>
        .
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          data_augmentation = tf.keras.Sequential([
          tf.keras.layers.RandomFlip(&#39;horizontal&#39;),
          tf.keras.layers.RandomRotation(0.2), ])
        </code>
      </pre>
      <aside className="note">
        <strong>Note:</strong>
        <span>
          These layers are active only during training, when you call
          <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit">
            <code translate="no" dir="ltr">
              Model.fit
            </code>
          </a>
          . They are inactive when the model is used in inference mode in
          <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate">
            <code translate="no" dir="ltr">
              Model.evaluate
            </code>
          </a>
          ,
          <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict">
            <code translate="no" dir="ltr">
              Model.predict
            </code>
          </a>
          , or
          <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model#call">
            <code translate="no" dir="ltr">
              Model.call
            </code>
          </a>
          .
        </span>
      </aside>
      <p>
        Let&#39;s repeatedly apply these layers to the same image and see the
        result.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          for image, _ in train_dataset.take(1): plt.figure(figsize=(10, 10))
          first_image = image[0] for i in range(9): ax = plt.subplot(3, 3, i +
          1) augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
          plt.imshow(augmented_image[0] / 255) plt.axis(&#39;off&#39;)
        </code>
      </pre>
      <p>
        <img
          src="/static/tutorials/images/transfer_learning_files/output_aQullOUHkm67_0.png"
          alt="png"
        />
      </p>

      <h3 id="rescale_pixel_values" data-text="Rescale pixel values">
        Rescale pixel values
      </h3>

      <p>
        In a moment, you will download
        <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet_v2/MobileNetV2">
          <code translate="no" dir="ltr">
            tf.keras.applications.MobileNetV2
          </code>
        </a>
        for use as your base model. This model expects pixel values in
        <code translate="no" dir="ltr">
          [-1, 1]
        </code>
        , but at this point, the pixel values in your images are in
        <code translate="no" dir="ltr">
          [0, 255]
        </code>
        . To rescale them, use the preprocessing method included with the model.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        </code>
      </pre>
      <aside className="note">
        <strong>Note:</strong>
        <span>
          Alternatively, you could rescale pixel values from
          <code translate="no" dir="ltr">
            [0, 255]
          </code>{" "}
          to
          <code translate="no" dir="ltr">
            [-1, 1]
          </code>{" "}
          using
          <code translate="no" dir="ltr">
            tf.keras.layers.Rescaling
          </code>
          .
        </span>
      </aside>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
        </code>
      </pre>
      <aside className="note">
        <strong>Note:</strong>
        <span>
          If using other
          <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications">
            <code translate="no" dir="ltr">
              tf.keras.applications
            </code>
          </a>
          , be sure to check the API doc to determine if they expect pixels in{" "}
          <code translate="no" dir="ltr">
            [-1, 1]
          </code>{" "}
          or
          <code translate="no" dir="ltr">
            [0, 1]
          </code>
          , or use the included
          <code translate="no" dir="ltr">
            preprocess_input
          </code>
          function.
        </span>
      </aside>
      <h2
        id="create_the_base_model_from_the_pre-trained_convnets"
        data-text="Create the base model from the pre-trained convnets"
      >
        Create the base model from the pre-trained convnets
      </h2>

      <p>
        You will create the base model from the
        <strong>MobileNet V2</strong> model developed at Google. This is
        pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M
        images and 1000 classNamees. ImageNet is a research training dataset
        with a wide variety of categories like{" "}
        <code translate="no" dir="ltr">
          jackfruit
        </code>{" "}
        and
        <code translate="no" dir="ltr">
          syringe
        </code>
        . This base of knowledge will help us classNameify cats and dogs from
        our specific dataset.
      </p>

      <p>
        First, you need to pick which layer of MobileNet V2 you will use for
        feature extraction. The very last classNameification layer (on
        &quot;top&quot;, as most diagrams of machine learning models go from
        bottom to top) is not very useful. Instead, you will follow the common
        practice to depend on the very last layer before the flatten operation.
        This layer is called the &quot;bottleneck layer&quot;. The bottleneck
        layer features retain more generality as compared to the final/top
        layer.
      </p>

      <p>
        First, instantiate a MobileNet V2 model pre-loaded with weights trained
        on ImageNet. By specifying the
        <strong>include_top=False</strong> argument, you load a network that
        doesn&#39;t include the classNameification layers at the top, which is
        ideal for feature extraction.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          # Create the base model from the pre-trained model MobileNet V2
          IMG_SHAPE = IMG_SIZE + (3,) base_model =
          tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
          include_top=False, weights=&#39;imagenet&#39;)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Downloading data from
        https&colon;//storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5
        9406464/9406464 [==============================] - 0s 0us/step
      </pre>

      <p>
        This feature extractor converts each
        <code translate="no" dir="ltr">
          160x160x3
        </code>{" "}
        image into a
        <code translate="no" dir="ltr">
          5x5x1280
        </code>{" "}
        block of features. Let&#39;s see what it does to an example batch of
        images:
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          image_batch, label_batch = next(iter(train_dataset)) feature_batch =
          base_model(image_batch) print(feature_batch.shape)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        (32, 5, 5, 1280)
      </pre>

      <h2 id="feature_extraction" data-text="Feature extraction">
        Feature extraction
      </h2>

      <p>
        In this step, you will freeze the convolutional base created from the
        previous step and to use as a feature extractor. Additionally, you add a
        classNameifier on top of it and train the top-level classNameifier.
      </p>

      <h3
        id="freeze_the_convolutional_base"
        data-text="Freeze the convolutional base"
      >
        Freeze the convolutional base
      </h3>

      <p>
        It is important to freeze the convolutional base before you compile and
        train the model. Freezing (by setting layer.trainable = False) prevents
        the weights in a given layer from being updated during training.
        MobileNet V2 has many layers, so setting the entire model&#39;s
        <code translate="no" dir="ltr">
          trainable
        </code>{" "}
        flag to False will freeze all of them.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          base_model.trainable = False
        </code>
      </pre>
      <h3
        id="important_note_about_batchnormalization_layers"
        data-text="Important note about BatchNormalization layers"
      >
        Important note about BatchNormalization layers
      </h3>

      <p>
        Many models contain
        <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization">
          <code translate="no" dir="ltr">
            tf.keras.layers.BatchNormalization
          </code>
        </a>
        layers. This layer is a special case and precautions should be taken in
        the context of fine-tuning, as shown later in this tutorial.
      </p>

      <p>
        When you set
        <code translate="no" dir="ltr">
          layer.trainable = False
        </code>
        , the
        <code translate="no" dir="ltr">
          BatchNormalization
        </code>{" "}
        layer will run in inference mode, and will not update its mean and
        variance statistics.
      </p>

      <p>
        When you unfreeze a model that contains BatchNormalization layers in
        order to do fine-tuning, you should keep the BatchNormalization layers
        in inference mode by passing
        <code translate="no" dir="ltr">
          training = False
        </code>{" "}
        when calling the base model. Otherwise, the updates applied to the
        non-trainable weights will destroy what the model has learned.
      </p>

      <p>
        For more details, see the
        <a href="https://www.tensorflow.org/guide/keras/transfer_learning">
          Transfer learning guide
        </a>
        .
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          # Let&#39;s take a look at the base model architecture
          base_model.summary()
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Model&colon; &quot;mobilenetv2_1.00_160&quot;
        __________________________________________________________________________________________________
        Layer (type) Output Shape Param # Connected to
        ==================================================================================================
        input_1 (InputLayer) [(None, 160, 160, 3)] 0 [] Conv1 (Conv2D) (None,
        80, 80, 32) 864 [&#x27;input_1[0][0]&#x27;] bn_Conv1 (BatchNormalizati
        (None, 80, 80, 32) 128 [&#x27;Conv1[0][0]&#x27;] on) Conv1_relu (ReLU)
        (None, 80, 80, 32) 0 [&#x27;bn_Conv1[0][0]&#x27;]
        expanded_conv_depthwise (D (None, 80, 80, 32) 288
        [&#x27;Conv1_relu[0][0]&#x27;] epthwiseConv2D)
        expanded_conv_depthwise_BN (None, 80, 80, 32) 128
        [&#x27;expanded_conv_depthwise[0][0 (BatchNormalization) ]&#x27;]
        expanded_conv_depthwise_re (None, 80, 80, 32) 0
        [&#x27;expanded_conv_depthwise_BN[0 lu (ReLU) ][0]&#x27;]
        expanded_conv_project (Con (None, 80, 80, 16) 512
        [&#x27;expanded_conv_depthwise_relu v2D) [0][0]&#x27;]
        expanded_conv_project_BN ( (None, 80, 80, 16) 64
        [&#x27;expanded_conv_project[0][0]&#x27; BatchNormalization) ]
        block_1_expand (Conv2D) (None, 80, 80, 96) 1536
        [&#x27;expanded_conv_project_BN[0][ 0]&#x27;] block_1_expand_BN (BatchNo
        (None, 80, 80, 96) 384 [&#x27;block_1_expand[0][0]&#x27;] rmalization)
        block_1_expand_relu (ReLU) (None, 80, 80, 96) 0
        [&#x27;block_1_expand_BN[0][0]&#x27;] block_1_pad (ZeroPadding2D (None,
        81, 81, 96) 0 [&#x27;block_1_expand_relu[0][0]&#x27;] )
        block_1_depthwise (Depthwi (None, 40, 40, 96) 864
        [&#x27;block_1_pad[0][0]&#x27;] seConv2D) block_1_depthwise_BN (Batc
        (None, 40, 40, 96) 384 [&#x27;block_1_depthwise[0][0]&#x27;]
        hNormalization) block_1_depthwise_relu (Re (None, 40, 40, 96) 0
        [&#x27;block_1_depthwise_BN[0][0]&#x27;] LU) block_1_project (Conv2D)
        (None, 40, 40, 24) 2304 [&#x27;block_1_depthwise_relu[0][0] &#x27;]
        block_1_project_BN (BatchN (None, 40, 40, 24) 96
        [&#x27;block_1_project[0][0]&#x27;] ormalization) block_2_expand
        (Conv2D) (None, 40, 40, 144) 3456 [&#x27;block_1_project_BN[0][0]&#x27;]
        block_2_expand_BN (BatchNo (None, 40, 40, 144) 576
        [&#x27;block_2_expand[0][0]&#x27;] rmalization) block_2_expand_relu
        (ReLU) (None, 40, 40, 144) 0 [&#x27;block_2_expand_BN[0][0]&#x27;]
        block_2_depthwise (Depthwi (None, 40, 40, 144) 1296
        [&#x27;block_2_expand_relu[0][0]&#x27;] seConv2D) block_2_depthwise_BN
        (Batc (None, 40, 40, 144) 576 [&#x27;block_2_depthwise[0][0]&#x27;]
        hNormalization) block_2_depthwise_relu (Re (None, 40, 40, 144) 0
        [&#x27;block_2_depthwise_BN[0][0]&#x27;] LU) block_2_project (Conv2D)
        (None, 40, 40, 24) 3456 [&#x27;block_2_depthwise_relu[0][0] &#x27;]
        block_2_project_BN (BatchN (None, 40, 40, 24) 96
        [&#x27;block_2_project[0][0]&#x27;] ormalization) block_2_add (Add)
        (None, 40, 40, 24) 0 [&#x27;block_1_project_BN[0][0]&#x27;,
        &#x27;block_2_project_BN[0][0]&#x27;] block_3_expand (Conv2D) (None, 40,
        40, 144) 3456 [&#x27;block_2_add[0][0]&#x27;] block_3_expand_BN (BatchNo
        (None, 40, 40, 144) 576 [&#x27;block_3_expand[0][0]&#x27;] rmalization)
        block_3_expand_relu (ReLU) (None, 40, 40, 144) 0
        [&#x27;block_3_expand_BN[0][0]&#x27;] block_3_pad (ZeroPadding2D (None,
        41, 41, 144) 0 [&#x27;block_3_expand_relu[0][0]&#x27;] )
        block_3_depthwise (Depthwi (None, 20, 20, 144) 1296
        [&#x27;block_3_pad[0][0]&#x27;] seConv2D) block_3_depthwise_BN (Batc
        (None, 20, 20, 144) 576 [&#x27;block_3_depthwise[0][0]&#x27;]
        hNormalization) block_3_depthwise_relu (Re (None, 20, 20, 144) 0
        [&#x27;block_3_depthwise_BN[0][0]&#x27;] LU) block_3_project (Conv2D)
        (None, 20, 20, 32) 4608 [&#x27;block_3_depthwise_relu[0][0] &#x27;]
        block_3_project_BN (BatchN (None, 20, 20, 32) 128
        [&#x27;block_3_project[0][0]&#x27;] ormalization) block_4_expand
        (Conv2D) (None, 20, 20, 192) 6144 [&#x27;block_3_project_BN[0][0]&#x27;]
        block_4_expand_BN (BatchNo (None, 20, 20, 192) 768
        [&#x27;block_4_expand[0][0]&#x27;] rmalization) block_4_expand_relu
        (ReLU) (None, 20, 20, 192) 0 [&#x27;block_4_expand_BN[0][0]&#x27;]
        block_4_depthwise (Depthwi (None, 20, 20, 192) 1728
        [&#x27;block_4_expand_relu[0][0]&#x27;] seConv2D) block_4_depthwise_BN
        (Batc (None, 20, 20, 192) 768 [&#x27;block_4_depthwise[0][0]&#x27;]
        hNormalization) block_4_depthwise_relu (Re (None, 20, 20, 192) 0
        [&#x27;block_4_depthwise_BN[0][0]&#x27;] LU) block_4_project (Conv2D)
        (None, 20, 20, 32) 6144 [&#x27;block_4_depthwise_relu[0][0] &#x27;]
        block_4_project_BN (BatchN (None, 20, 20, 32) 128
        [&#x27;block_4_project[0][0]&#x27;] ormalization) block_4_add (Add)
        (None, 20, 20, 32) 0 [&#x27;block_3_project_BN[0][0]&#x27;,
        &#x27;block_4_project_BN[0][0]&#x27;] block_5_expand (Conv2D) (None, 20,
        20, 192) 6144 [&#x27;block_4_add[0][0]&#x27;] block_5_expand_BN (BatchNo
        (None, 20, 20, 192) 768 [&#x27;block_5_expand[0][0]&#x27;] rmalization)
        block_5_expand_relu (ReLU) (None, 20, 20, 192) 0
        [&#x27;block_5_expand_BN[0][0]&#x27;] block_5_depthwise (Depthwi (None,
        20, 20, 192) 1728 [&#x27;block_5_expand_relu[0][0]&#x27;] seConv2D)
        block_5_depthwise_BN (Batc (None, 20, 20, 192) 768
        [&#x27;block_5_depthwise[0][0]&#x27;] hNormalization)
        block_5_depthwise_relu (Re (None, 20, 20, 192) 0
        [&#x27;block_5_depthwise_BN[0][0]&#x27;] LU) block_5_project (Conv2D)
        (None, 20, 20, 32) 6144 [&#x27;block_5_depthwise_relu[0][0] &#x27;]
        block_5_project_BN (BatchN (None, 20, 20, 32) 128
        [&#x27;block_5_project[0][0]&#x27;] ormalization) block_5_add (Add)
        (None, 20, 20, 32) 0 [&#x27;block_4_add[0][0]&#x27;,
        &#x27;block_5_project_BN[0][0]&#x27;] block_6_expand (Conv2D) (None, 20,
        20, 192) 6144 [&#x27;block_5_add[0][0]&#x27;] block_6_expand_BN (BatchNo
        (None, 20, 20, 192) 768 [&#x27;block_6_expand[0][0]&#x27;] rmalization)
        block_6_expand_relu (ReLU) (None, 20, 20, 192) 0
        [&#x27;block_6_expand_BN[0][0]&#x27;] block_6_pad (ZeroPadding2D (None,
        21, 21, 192) 0 [&#x27;block_6_expand_relu[0][0]&#x27;] )
        block_6_depthwise (Depthwi (None, 10, 10, 192) 1728
        [&#x27;block_6_pad[0][0]&#x27;] seConv2D) block_6_depthwise_BN (Batc
        (None, 10, 10, 192) 768 [&#x27;block_6_depthwise[0][0]&#x27;]
        hNormalization) block_6_depthwise_relu (Re (None, 10, 10, 192) 0
        [&#x27;block_6_depthwise_BN[0][0]&#x27;] LU) block_6_project (Conv2D)
        (None, 10, 10, 64) 12288 [&#x27;block_6_depthwise_relu[0][0] &#x27;]
        block_6_project_BN (BatchN (None, 10, 10, 64) 256
        [&#x27;block_6_project[0][0]&#x27;] ormalization) block_7_expand
        (Conv2D) (None, 10, 10, 384) 24576
        [&#x27;block_6_project_BN[0][0]&#x27;] block_7_expand_BN (BatchNo (None,
        10, 10, 384) 1536 [&#x27;block_7_expand[0][0]&#x27;] rmalization)
        block_7_expand_relu (ReLU) (None, 10, 10, 384) 0
        [&#x27;block_7_expand_BN[0][0]&#x27;] block_7_depthwise (Depthwi (None,
        10, 10, 384) 3456 [&#x27;block_7_expand_relu[0][0]&#x27;] seConv2D)
        block_7_depthwise_BN (Batc (None, 10, 10, 384) 1536
        [&#x27;block_7_depthwise[0][0]&#x27;] hNormalization)
        block_7_depthwise_relu (Re (None, 10, 10, 384) 0
        [&#x27;block_7_depthwise_BN[0][0]&#x27;] LU) block_7_project (Conv2D)
        (None, 10, 10, 64) 24576 [&#x27;block_7_depthwise_relu[0][0] &#x27;]
        block_7_project_BN (BatchN (None, 10, 10, 64) 256
        [&#x27;block_7_project[0][0]&#x27;] ormalization) block_7_add (Add)
        (None, 10, 10, 64) 0 [&#x27;block_6_project_BN[0][0]&#x27;,
        &#x27;block_7_project_BN[0][0]&#x27;] block_8_expand (Conv2D) (None, 10,
        10, 384) 24576 [&#x27;block_7_add[0][0]&#x27;] block_8_expand_BN
        (BatchNo (None, 10, 10, 384) 1536 [&#x27;block_8_expand[0][0]&#x27;]
        rmalization) block_8_expand_relu (ReLU) (None, 10, 10, 384) 0
        [&#x27;block_8_expand_BN[0][0]&#x27;] block_8_depthwise (Depthwi (None,
        10, 10, 384) 3456 [&#x27;block_8_expand_relu[0][0]&#x27;] seConv2D)
        block_8_depthwise_BN (Batc (None, 10, 10, 384) 1536
        [&#x27;block_8_depthwise[0][0]&#x27;] hNormalization)
        block_8_depthwise_relu (Re (None, 10, 10, 384) 0
        [&#x27;block_8_depthwise_BN[0][0]&#x27;] LU) block_8_project (Conv2D)
        (None, 10, 10, 64) 24576 [&#x27;block_8_depthwise_relu[0][0] &#x27;]
        block_8_project_BN (BatchN (None, 10, 10, 64) 256
        [&#x27;block_8_project[0][0]&#x27;] ormalization) block_8_add (Add)
        (None, 10, 10, 64) 0 [&#x27;block_7_add[0][0]&#x27;,
        &#x27;block_8_project_BN[0][0]&#x27;] block_9_expand (Conv2D) (None, 10,
        10, 384) 24576 [&#x27;block_8_add[0][0]&#x27;] block_9_expand_BN
        (BatchNo (None, 10, 10, 384) 1536 [&#x27;block_9_expand[0][0]&#x27;]
        rmalization) block_9_expand_relu (ReLU) (None, 10, 10, 384) 0
        [&#x27;block_9_expand_BN[0][0]&#x27;] block_9_depthwise (Depthwi (None,
        10, 10, 384) 3456 [&#x27;block_9_expand_relu[0][0]&#x27;] seConv2D)
        block_9_depthwise_BN (Batc (None, 10, 10, 384) 1536
        [&#x27;block_9_depthwise[0][0]&#x27;] hNormalization)
        block_9_depthwise_relu (Re (None, 10, 10, 384) 0
        [&#x27;block_9_depthwise_BN[0][0]&#x27;] LU) block_9_project (Conv2D)
        (None, 10, 10, 64) 24576 [&#x27;block_9_depthwise_relu[0][0] &#x27;]
        block_9_project_BN (BatchN (None, 10, 10, 64) 256
        [&#x27;block_9_project[0][0]&#x27;] ormalization) block_9_add (Add)
        (None, 10, 10, 64) 0 [&#x27;block_8_add[0][0]&#x27;,
        &#x27;block_9_project_BN[0][0]&#x27;] block_10_expand (Conv2D) (None,
        10, 10, 384) 24576 [&#x27;block_9_add[0][0]&#x27;] block_10_expand_BN
        (BatchN (None, 10, 10, 384) 1536 [&#x27;block_10_expand[0][0]&#x27;]
        ormalization) block_10_expand_relu (ReLU (None, 10, 10, 384) 0
        [&#x27;block_10_expand_BN[0][0]&#x27;] ) block_10_depthwise (Depthw
        (None, 10, 10, 384) 3456 [&#x27;block_10_expand_relu[0][0]&#x27;]
        iseConv2D) block_10_depthwise_BN (Bat (None, 10, 10, 384) 1536
        [&#x27;block_10_depthwise[0][0]&#x27;] chNormalization)
        block_10_depthwise_relu (R (None, 10, 10, 384) 0
        [&#x27;block_10_depthwise_BN[0][0]&#x27; eLU) ] block_10_project
        (Conv2D) (None, 10, 10, 96) 36864 [&#x27;block_10_depthwise_relu[0][0
        ]&#x27;] block_10_project_BN (Batch (None, 10, 10, 96) 384
        [&#x27;block_10_project[0][0]&#x27;] Normalization) block_11_expand
        (Conv2D) (None, 10, 10, 576) 55296
        [&#x27;block_10_project_BN[0][0]&#x27;] block_11_expand_BN (BatchN
        (None, 10, 10, 576) 2304 [&#x27;block_11_expand[0][0]&#x27;]
        ormalization) block_11_expand_relu (ReLU (None, 10, 10, 576) 0
        [&#x27;block_11_expand_BN[0][0]&#x27;] ) block_11_depthwise (Depthw
        (None, 10, 10, 576) 5184 [&#x27;block_11_expand_relu[0][0]&#x27;]
        iseConv2D) block_11_depthwise_BN (Bat (None, 10, 10, 576) 2304
        [&#x27;block_11_depthwise[0][0]&#x27;] chNormalization)
        block_11_depthwise_relu (R (None, 10, 10, 576) 0
        [&#x27;block_11_depthwise_BN[0][0]&#x27; eLU) ] block_11_project
        (Conv2D) (None, 10, 10, 96) 55296 [&#x27;block_11_depthwise_relu[0][0
        ]&#x27;] block_11_project_BN (Batch (None, 10, 10, 96) 384
        [&#x27;block_11_project[0][0]&#x27;] Normalization) block_11_add (Add)
        (None, 10, 10, 96) 0 [&#x27;block_10_project_BN[0][0]&#x27;,
        &#x27;block_11_project_BN[0][0]&#x27;] block_12_expand (Conv2D) (None,
        10, 10, 576) 55296 [&#x27;block_11_add[0][0]&#x27;] block_12_expand_BN
        (BatchN (None, 10, 10, 576) 2304 [&#x27;block_12_expand[0][0]&#x27;]
        ormalization) block_12_expand_relu (ReLU (None, 10, 10, 576) 0
        [&#x27;block_12_expand_BN[0][0]&#x27;] ) block_12_depthwise (Depthw
        (None, 10, 10, 576) 5184 [&#x27;block_12_expand_relu[0][0]&#x27;]
        iseConv2D) block_12_depthwise_BN (Bat (None, 10, 10, 576) 2304
        [&#x27;block_12_depthwise[0][0]&#x27;] chNormalization)
        block_12_depthwise_relu (R (None, 10, 10, 576) 0
        [&#x27;block_12_depthwise_BN[0][0]&#x27; eLU) ] block_12_project
        (Conv2D) (None, 10, 10, 96) 55296 [&#x27;block_12_depthwise_relu[0][0
        ]&#x27;] block_12_project_BN (Batch (None, 10, 10, 96) 384
        [&#x27;block_12_project[0][0]&#x27;] Normalization) block_12_add (Add)
        (None, 10, 10, 96) 0 [&#x27;block_11_add[0][0]&#x27;,
        &#x27;block_12_project_BN[0][0]&#x27;] block_13_expand (Conv2D) (None,
        10, 10, 576) 55296 [&#x27;block_12_add[0][0]&#x27;] block_13_expand_BN
        (BatchN (None, 10, 10, 576) 2304 [&#x27;block_13_expand[0][0]&#x27;]
        ormalization) block_13_expand_relu (ReLU (None, 10, 10, 576) 0
        [&#x27;block_13_expand_BN[0][0]&#x27;] ) block_13_pad (ZeroPadding2
        (None, 11, 11, 576) 0 [&#x27;block_13_expand_relu[0][0]&#x27;] D)
        block_13_depthwise (Depthw (None, 5, 5, 576) 5184
        [&#x27;block_13_pad[0][0]&#x27;] iseConv2D) block_13_depthwise_BN (Bat
        (None, 5, 5, 576) 2304 [&#x27;block_13_depthwise[0][0]&#x27;]
        chNormalization) block_13_depthwise_relu (R (None, 5, 5, 576) 0
        [&#x27;block_13_depthwise_BN[0][0]&#x27; eLU) ] block_13_project
        (Conv2D) (None, 5, 5, 160) 92160 [&#x27;block_13_depthwise_relu[0][0
        ]&#x27;] block_13_project_BN (Batch (None, 5, 5, 160) 640
        [&#x27;block_13_project[0][0]&#x27;] Normalization) block_14_expand
        (Conv2D) (None, 5, 5, 960) 153600
        [&#x27;block_13_project_BN[0][0]&#x27;] block_14_expand_BN (BatchN
        (None, 5, 5, 960) 3840 [&#x27;block_14_expand[0][0]&#x27;] ormalization)
        block_14_expand_relu (ReLU (None, 5, 5, 960) 0
        [&#x27;block_14_expand_BN[0][0]&#x27;] ) block_14_depthwise (Depthw
        (None, 5, 5, 960) 8640 [&#x27;block_14_expand_relu[0][0]&#x27;]
        iseConv2D) block_14_depthwise_BN (Bat (None, 5, 5, 960) 3840
        [&#x27;block_14_depthwise[0][0]&#x27;] chNormalization)
        block_14_depthwise_relu (R (None, 5, 5, 960) 0
        [&#x27;block_14_depthwise_BN[0][0]&#x27; eLU) ] block_14_project
        (Conv2D) (None, 5, 5, 160) 153600 [&#x27;block_14_depthwise_relu[0][0
        ]&#x27;] block_14_project_BN (Batch (None, 5, 5, 160) 640
        [&#x27;block_14_project[0][0]&#x27;] Normalization) block_14_add (Add)
        (None, 5, 5, 160) 0 [&#x27;block_13_project_BN[0][0]&#x27;,
        &#x27;block_14_project_BN[0][0]&#x27;] block_15_expand (Conv2D) (None,
        5, 5, 960) 153600 [&#x27;block_14_add[0][0]&#x27;] block_15_expand_BN
        (BatchN (None, 5, 5, 960) 3840 [&#x27;block_15_expand[0][0]&#x27;]
        ormalization) block_15_expand_relu (ReLU (None, 5, 5, 960) 0
        [&#x27;block_15_expand_BN[0][0]&#x27;] ) block_15_depthwise (Depthw
        (None, 5, 5, 960) 8640 [&#x27;block_15_expand_relu[0][0]&#x27;]
        iseConv2D) block_15_depthwise_BN (Bat (None, 5, 5, 960) 3840
        [&#x27;block_15_depthwise[0][0]&#x27;] chNormalization)
        block_15_depthwise_relu (R (None, 5, 5, 960) 0
        [&#x27;block_15_depthwise_BN[0][0]&#x27; eLU) ] block_15_project
        (Conv2D) (None, 5, 5, 160) 153600 [&#x27;block_15_depthwise_relu[0][0
        ]&#x27;] block_15_project_BN (Batch (None, 5, 5, 160) 640
        [&#x27;block_15_project[0][0]&#x27;] Normalization) block_15_add (Add)
        (None, 5, 5, 160) 0 [&#x27;block_14_add[0][0]&#x27;,
        &#x27;block_15_project_BN[0][0]&#x27;] block_16_expand (Conv2D) (None,
        5, 5, 960) 153600 [&#x27;block_15_add[0][0]&#x27;] block_16_expand_BN
        (BatchN (None, 5, 5, 960) 3840 [&#x27;block_16_expand[0][0]&#x27;]
        ormalization) block_16_expand_relu (ReLU (None, 5, 5, 960) 0
        [&#x27;block_16_expand_BN[0][0]&#x27;] ) block_16_depthwise (Depthw
        (None, 5, 5, 960) 8640 [&#x27;block_16_expand_relu[0][0]&#x27;]
        iseConv2D) block_16_depthwise_BN (Bat (None, 5, 5, 960) 3840
        [&#x27;block_16_depthwise[0][0]&#x27;] chNormalization)
        block_16_depthwise_relu (R (None, 5, 5, 960) 0
        [&#x27;block_16_depthwise_BN[0][0]&#x27; eLU) ] block_16_project
        (Conv2D) (None, 5, 5, 320) 307200 [&#x27;block_16_depthwise_relu[0][0
        ]&#x27;] block_16_project_BN (Batch (None, 5, 5, 320) 1280
        [&#x27;block_16_project[0][0]&#x27;] Normalization) Conv_1 (Conv2D)
        (None, 5, 5, 1280) 409600 [&#x27;block_16_project_BN[0][0]&#x27;]
        Conv_1_bn (BatchNormalizat (None, 5, 5, 1280) 5120
        [&#x27;Conv_1[0][0]&#x27;] ion) out_relu (ReLU) (None, 5, 5, 1280) 0
        [&#x27;Conv_1_bn[0][0]&#x27;]
        ==================================================================================================
        Total params&colon; 2257984 (8.61 MB) Trainable params&colon; 0 (0.00
        Byte) Non-trainable params&colon; 2257984 (8.61 MB)
        __________________________________________________________________________________________________
      </pre>

      <h3
        id="add_a_classNameification_head"
        data-text="Add a classNameification head"
      >
        Add a classNameification head
      </h3>

      <p>
        To generate predictions from the block of features, average over the
        spatial
        <code translate="no" dir="ltr">
          5x5
        </code>{" "}
        spatial locations, using a
        <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling2D">
          <code translate="no" dir="ltr">
            tf.keras.layers.GlobalAveragePooling2D
          </code>
        </a>
        layer to convert the features to a single 1280-element vector per image.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
          feature_batch_average = global_average_layer(feature_batch)
          print(feature_batch_average.shape)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        (32, 1280)
      </pre>

      <p>
        Apply a
        <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense">
          <code translate="no" dir="ltr">
            tf.keras.layers.Dense
          </code>
        </a>
        layer to convert these features into a single prediction per image. You
        don&#39;t need an activation function here because this prediction will
        be treated as a
        <code translate="no" dir="ltr">
          logit
        </code>
        , or a raw prediction value. Positive numbers predict className 1,
        negative numbers predict className 0.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          prediction_layer = tf.keras.layers.Dense(1) prediction_batch =
          prediction_layer(feature_batch_average) print(prediction_batch.shape)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        (32, 1)
      </pre>

      <p>
        Build a model by chaining together the data augmentation, rescaling,
        <code translate="no" dir="ltr">
          base_model
        </code>{" "}
        and feature extractor layers using the
        <a href="https://www.tensorflow.org/guide/keras/functional">
          Keras Functional API
        </a>
        . As previously mentioned, use
        <code translate="no" dir="ltr">
          training=False
        </code>{" "}
        as our model contains a
        <code translate="no" dir="ltr">
          BatchNormalization
        </code>
        layer.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          inputs = tf.keras.Input(shape=(160, 160, 3)) x =
          data_augmentation(inputs) x = preprocess_input(x) x = base_model(x,
          training=False) x = global_average_layer(x) x =
          tf.keras.layers.Dropout(0.2)(x) outputs = prediction_layer(x) model =
          tf.keras.Model(inputs, outputs)
        </code>
      </pre>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          model.summary()
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Model&colon; &quot;model&quot;
        _________________________________________________________________ Layer
        (type) Output Shape Param #
        =================================================================
        input_2 (InputLayer) [(None, 160, 160, 3)] 0 sequential (Sequential)
        (None, 160, 160, 3) 0 tf.math.truediv (TFOpLambd (None, 160, 160, 3) 0
        a) tf.math.subtract (TFOpLamb (None, 160, 160, 3) 0 da)
        mobilenetv2_1.00_160 (Func (None, 5, 5, 1280) 2257984 tional)
        global_average_pooling2d ( (None, 1280) 0 GlobalAveragePooling2D)
        dropout (Dropout) (None, 1280) 0 dense (Dense) (None, 1) 1281
        ================================================================= Total
        params&colon; 2259265 (8.62 MB) Trainable params&colon; 1281 (5.00 KB)
        Non-trainable params&colon; 2257984 (8.61 MB)
        _________________________________________________________________
      </pre>

      <p>
        The 8+ million parameters in MobileNet are frozen, but there are 1.2
        thousand <em>trainable</em> parameters in the Dense layer. These are
        divided between two
        <a href="https://www.tensorflow.org/api_docs/python/tf/Variable">
          <code translate="no" dir="ltr">
            tf.Variable
          </code>
        </a>
        objects, the weights and biases.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          len(model.trainable_variables)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        2
      </pre>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          tf.keras.utils.plot_model(model, show_shapes=True)
        </code>
      </pre>
      <p>
        <img
          src="/static/tutorials/images/transfer_learning_files/output_jeGk93R2ahav_0.png"
          alt="png"
        />
      </p>

      <h3 id="compile_the_model" data-text="Compile the model">
        Compile the model
      </h3>

      <p>
        Compile the model before training it. Since there are two classNamees,
        use the
        <a href="https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy">
          <code translate="no" dir="ltr">
            tf.keras.losses.BinaryCrossentropy
          </code>
        </a>
        loss with
        <code translate="no" dir="ltr">
          from_logits=True
        </code>{" "}
        since the model provides a linear output.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          base_learning_rate = 0.0001
          model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
          loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
          metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0,
          name=&#39;accuracy&#39;)])
        </code>
      </pre>
      <h3 id="train_the_model" data-text="Train the model">
        Train the model
      </h3>

      <p>
        After training for 10 epochs, you should see ~96% accuracy on the
        validation set.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          initial_epochs = 10 loss0, accuracy0 =
          model.evaluate(validation_dataset)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        26/26 [==============================] - 3s 38ms/step - loss&colon;
        1.0864 - accuracy&colon; 0.2228
      </pre>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          {'print("initial loss: {:.2f}".format(loss0))'}
          {'print("initial accuracy: {:.2f}".format(accuracy0))'}
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        initial loss&colon; 1.09 initial accuracy&colon; 0.22
      </pre>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          history = model.fit(train_dataset, epochs=initial_epochs,
          validation_data=validation_dataset)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Epoch 1/10 63/63 [==============================] - 6s 45ms/step -
        loss&colon; 0.9132 - accuracy&colon; 0.4025 - val_loss&colon; 0.7343 -
        val_accuracy&colon; 0.5136 Epoch 2/10 63/63
        [==============================] - 2s 38ms/step - loss&colon; 0.6518 -
        accuracy&colon; 0.6340 - val_loss&colon; 0.5089 - val_accuracy&colon;
        0.7847 Epoch 3/10 63/63 [==============================] - 2s 37ms/step
        - loss&colon; 0.4936 - accuracy&colon; 0.7685 - val_loss&colon; 0.3806 -
        val_accuracy&colon; 0.8948 Epoch 4/10 63/63
        [==============================] - 2s 37ms/step - loss&colon; 0.3950 -
        accuracy&colon; 0.8440 - val_loss&colon; 0.3052 - val_accuracy&colon;
        0.9307 Epoch 5/10 63/63 [==============================] - 2s 37ms/step
        - loss&colon; 0.3506 - accuracy&colon; 0.8645 - val_loss&colon; 0.2457 -
        val_accuracy&colon; 0.9567 Epoch 6/10 63/63
        [==============================] - 2s 38ms/step - loss&colon; 0.3016 -
        accuracy&colon; 0.8915 - val_loss&colon; 0.2156 - val_accuracy&colon;
        0.9592 Epoch 7/10 63/63 [==============================] - 2s 37ms/step
        - loss&colon; 0.2725 - accuracy&colon; 0.9055 - val_loss&colon; 0.1880 -
        val_accuracy&colon; 0.9629 Epoch 8/10 63/63
        [==============================] - 2s 37ms/step - loss&colon; 0.2567 -
        accuracy&colon; 0.9100 - val_loss&colon; 0.1705 - val_accuracy&colon;
        0.9592 Epoch 9/10 63/63 [==============================] - 2s 37ms/step
        - loss&colon; 0.2394 - accuracy&colon; 0.9120 - val_loss&colon; 0.1543 -
        val_accuracy&colon; 0.9691 Epoch 10/10 63/63
        [==============================] - 2s 38ms/step - loss&colon; 0.2243 -
        accuracy&colon; 0.9170 - val_loss&colon; 0.1387 - val_accuracy&colon;
        0.9678
      </pre>

      <h3 id="learning_curves" data-text="Learning curves">
        Learning curves
      </h3>

      <p>
        Let&#39;s take a look at the learning curves of the training and
        validation accuracy/loss when using the MobileNetV2 base model as a
        fixed feature extractor.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          acc = history.history[&#39;accuracy&#39;] val_acc =
          history.history[&#39;val_accuracy&#39;] loss =
          history.history[&#39;loss&#39;] val_loss =
          history.history[&#39;val_loss&#39;] plt.figure(figsize=(8, 8))
          plt.subplot(2, 1, 1) plt.plot(acc, label=&#39;Training Accuracy&#39;)
          plt.plot(val_acc, label=&#39;Validation Accuracy&#39;)
          plt.legend(loc=&#39;lower right&#39;) plt.ylabel(&#39;Accuracy&#39;)
          plt.ylim([min(plt.ylim()),1]) plt.title(&#39;Training and Validation
          Accuracy&#39;) plt.subplot(2, 1, 2) plt.plot(loss, label=&#39;Training
          Loss&#39;) plt.plot(val_loss, label=&#39;Validation Loss&#39;)
          plt.legend(loc=&#39;upper right&#39;) plt.ylabel(&#39;Cross
          Entropy&#39;) plt.ylim([0,1.0]) plt.title(&#39;Training and Validation
          Loss&#39;) plt.xlabel(&#39;epoch&#39;) plt.show()
        </code>
      </pre>
      <p>
        <img
          src="/static/tutorials/images/transfer_learning_files/output_53OTCh3jnbwV_0.png"
          alt="png"
        />
      </p>
      <aside className="note">
        <strong>Note:</strong>
        <span>
          If you are wondering why the validation metrics are clearly better
          than the training metrics, the main factor is because layers like
          <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization">
            <code translate="no" dir="ltr">
              tf.keras.layers.BatchNormalization
            </code>
          </a>
          and
          <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout">
            <code translate="no" dir="ltr">
              tf.keras.layers.Dropout
            </code>
          </a>
          affect accuracy during training. They are turned off when calculating
          validation loss.
        </span>
      </aside>
      <p>
        To a lesser extent, it is also because training metrics report the
        average for an epoch, while validation metrics are evaluated after the
        epoch, so validation metrics see a model that has trained slightly
        longer.
      </p>

      <h2 id="fine_tuning" data-text="Fine tuning">
        Fine tuning
      </h2>

      <p>
        In the feature extraction experiment, you were only training a few
        layers on top of an MobileNetV2 base model. The weights of the
        pre-trained network were <strong>not</strong> updated during training.
      </p>

      <p>
        One way to increase performance even further is to train (or
        &quot;fine-tune&quot;) the weights of the top layers of the pre-trained
        model alongside the training of the classNameifier you added. The
        training process will force the weights to be tuned from generic feature
        maps to features associated specifically with the dataset.
      </p>
      <aside className="note">
        <strong>Note:</strong>
        <span>
          This should only be attempted after you have trained the top-level
          classNameifier with the pre-trained model set to non-trainable. If you
          add a randomly initialized classNameifier on top of a pre-trained
          model and attempt to train all layers jointly, the magnitude of the
          gradient updates will be too large (due to the random weights from the
          classNameifier) and your pre-trained model will forget what it has
          learned.
        </span>
      </aside>
      <p>
        Also, you should try to fine-tune a small number of top layers rather
        than the whole MobileNet model. In most convolutional networks, the
        higher up a layer is, the more specialized it is. The first few layers
        learn very simple and generic features that generalize to almost all
        types of images. As you go higher up, the features are increasingly more
        specific to the dataset on which the model was trained. The goal of
        fine-tuning is to adapt these specialized features to work with the new
        dataset, rather than overwrite the generic learning.
      </p>

      <h3
        id="un-freeze_the_top_layers_of_the_model"
        data-text="Un-freeze the top layers of the model"
      >
        Un-freeze the top layers of the model
      </h3>

      <p>
        All you need to do is unfreeze the
        <code translate="no" dir="ltr">
          base_model
        </code>{" "}
        and set the bottom layers to be un-trainable. Then, you should recompile
        the model (necessary for these changes to take effect), and resume
        training.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          base_model.trainable = True
        </code>
      </pre>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          # Let&#39;s take a look to see how many layers are in the base model
          print(&#34;Number of layers in the base model: &#34;,
          len(base_model.layers)) # Fine-tune from this layer onwards
          fine_tune_at = 100 # Freeze all the layers before the `fine_tune_at`
          layer for layer in base_model.layers[:fine_tune_at]: layer.trainable =
          False
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Number of layers in the base model&colon; 154
      </pre>

      <h3 id="compile_the_model_2" data-text="Compile the model">
        Compile the model
      </h3>

      <p>
        As you are training a much larger model and want to readapt the
        pretrained weights, it is important to use a lower learning rate at this
        stage. Otherwise, your model could overfit very quickly.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
          optimizer =
          tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
          metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0,
          name=&#39;accuracy&#39;)])
        </code>
      </pre>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          model.summary()
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Model&colon; &quot;model&quot;
        _________________________________________________________________ Layer
        (type) Output Shape Param #
        =================================================================
        input_2 (InputLayer) [(None, 160, 160, 3)] 0 sequential (Sequential)
        (None, 160, 160, 3) 0 tf.math.truediv (TFOpLambd (None, 160, 160, 3) 0
        a) tf.math.subtract (TFOpLamb (None, 160, 160, 3) 0 da)
        mobilenetv2_1.00_160 (Func (None, 5, 5, 1280) 2257984 tional)
        global_average_pooling2d ( (None, 1280) 0 GlobalAveragePooling2D)
        dropout (Dropout) (None, 1280) 0 dense (Dense) (None, 1) 1281
        ================================================================= Total
        params&colon; 2259265 (8.62 MB) Trainable params&colon; 1862721 (7.11
        MB) Non-trainable params&colon; 396544 (1.51 MB)
        _________________________________________________________________
      </pre>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          len(model.trainable_variables)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        56
      </pre>

      <h3
        id="continue_training_the_model"
        data-text="Continue training the model"
      >
        Continue training the model
      </h3>

      <p>
        If you trained to convergence earlier, this step will improve your
        accuracy by a few percentage points.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          fine_tune_epochs = 10 total_epochs = initial_epochs + fine_tune_epochs
          history_fine = model.fit(train_dataset, epochs=total_epochs,
          initial_epoch=history.epoch[-1], validation_data=validation_dataset)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Epoch 10/20 63/63 [==============================] - 11s 64ms/step -
        loss&colon; 0.1555 - accuracy&colon; 0.9365 - val_loss&colon; 0.0543 -
        val_accuracy&colon; 0.9827 Epoch 11/20 63/63
        [==============================] - 3s 47ms/step - loss&colon; 0.1151 -
        accuracy&colon; 0.9500 - val_loss&colon; 0.0739 - val_accuracy&colon;
        0.9790 Epoch 12/20 63/63 [==============================] - 3s 47ms/step
        - loss&colon; 0.1042 - accuracy&colon; 0.9630 - val_loss&colon; 0.0415 -
        val_accuracy&colon; 0.9851 Epoch 13/20 63/63
        [==============================] - 3s 47ms/step - loss&colon; 0.1043 -
        accuracy&colon; 0.9610 - val_loss&colon; 0.0362 - val_accuracy&colon;
        0.9889 Epoch 14/20 63/63 [==============================] - 3s 47ms/step
        - loss&colon; 0.0814 - accuracy&colon; 0.9650 - val_loss&colon; 0.0371 -
        val_accuracy&colon; 0.9864 Epoch 15/20 63/63
        [==============================] - 3s 48ms/step - loss&colon; 0.0770 -
        accuracy&colon; 0.9670 - val_loss&colon; 0.0417 - val_accuracy&colon;
        0.9839 Epoch 16/20 63/63 [==============================] - 3s 48ms/step
        - loss&colon; 0.0719 - accuracy&colon; 0.9735 - val_loss&colon; 0.0363 -
        val_accuracy&colon; 0.9839 Epoch 17/20 63/63
        [==============================] - 3s 48ms/step - loss&colon; 0.0768 -
        accuracy&colon; 0.9680 - val_loss&colon; 0.0397 - val_accuracy&colon;
        0.9851 Epoch 18/20 63/63 [==============================] - 3s 48ms/step
        - loss&colon; 0.0736 - accuracy&colon; 0.9715 - val_loss&colon; 0.0347 -
        val_accuracy&colon; 0.9889 Epoch 19/20 63/63
        [==============================] - 3s 48ms/step - loss&colon; 0.0510 -
        accuracy&colon; 0.9810 - val_loss&colon; 0.0373 - val_accuracy&colon;
        0.9839 Epoch 20/20 63/63 [==============================] - 3s 48ms/step
        - loss&colon; 0.0552 - accuracy&colon; 0.9815 - val_loss&colon; 0.0345 -
        val_accuracy&colon; 0.9889
      </pre>

      <p>
        Let&#39;s take a look at the learning curves of the training and
        validation accuracy/loss when fine-tuning the last few layers of the
        MobileNetV2 base model and training the classNameifier on top of it. The
        validation loss is much higher than the training loss, so you may get
        some overfitting.
      </p>

      <p>
        You may also get some overfitting as the new training set is relatively
        small and similar to the original MobileNetV2 datasets.
      </p>

      <p>
        After fine tuning the model nearly reaches 98% accuracy on the
        validation set.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          acc += history_fine.history[&#39;accuracy&#39;] val_acc +=
          history_fine.history[&#39;val_accuracy&#39;] loss +=
          history_fine.history[&#39;loss&#39;] val_loss +=
          history_fine.history[&#39;val_loss&#39;]
        </code>
      </pre>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          plt.figure(figsize=(8, 8)) plt.subplot(2, 1, 1) plt.plot(acc,
          label=&#39;Training Accuracy&#39;) plt.plot(val_acc,
          label=&#39;Validation Accuracy&#39;) plt.ylim([0.8, 1])
          plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(),
          label=&#39;Start Fine Tuning&#39;) plt.legend(loc=&#39;lower
          right&#39;) plt.title(&#39;Training and Validation Accuracy&#39;)
          plt.subplot(2, 1, 2) plt.plot(loss, label=&#39;Training Loss&#39;)
          plt.plot(val_loss, label=&#39;Validation Loss&#39;) plt.ylim([0, 1.0])
          plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(),
          label=&#39;Start Fine Tuning&#39;) plt.legend(loc=&#39;upper
          right&#39;) plt.title(&#39;Training and Validation Loss&#39;)
          plt.xlabel(&#39;epoch&#39;) plt.show()
        </code>
      </pre>
      <p>
        <img
          src="/static/tutorials/images/transfer_learning_files/output_chW103JUItdk_0.png"
          alt="png"
        />
      </p>

      <h3 id="evaluation_and_prediction" data-text="Evaluation and prediction">
        Evaluation and prediction
      </h3>

      <p>
        Finally you can verify the performance of the model on new data using
        test set.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          loss, accuracy = model.evaluate(test_dataset) print(&#39;Test accuracy
          :&#39;, accuracy)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        6/6 [==============================] - 0s 26ms/step - loss&colon; 0.0262
        - accuracy&colon; 0.9948 Test accuracy &colon; 0.9947916865348816
      </pre>

      <p>
        And now you are all set to use this model to predict if your pet is a
        cat or dog.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          # Retrieve a batch of images from the test set image_batch,
          label_batch = test_dataset.as_numpy_iterator().next() predictions =
          model.predict_on_batch(image_batch).flatten() # Apply a sigmoid since
          our model returns logits predictions = tf.nn.sigmoid(predictions)
          predictions = tf.where(predictions &lt; 0.5, 0, 1)
          print(&#39;Predictions:\n&#39;, predictions.numpy())
          print(&#39;Labels:\n&#39;, label_batch) plt.figure(figsize=(10, 10))
          for i in range(9): ax = plt.subplot(3, 3, i + 1)
          plt.imshow(image_batch[i].astype(&#34;uint8&#34;))
          plt.title(className_names[predictions[i]]) plt.axis(&#34;off&#34;)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Predictions&colon; [0 1 0 1 1 1 0 1 1 0 1 0 0 0 0 1 1 0 1 0 1 1 1 1 0 1
        0 1 0 1 1 0] Labels&colon; [0 1 0 1 1 1 0 1 1 0 1 0 0 0 0 1 1 0 1 0 1 1
        1 1 0 1 0 1 0 1 1 0]
      </pre>

      <p>
        <img
          src="/static/tutorials/images/transfer_learning_files/output_RUNoQNgtfNgt_1.png"
          alt="png"
        />
      </p>

      <h2 id="summary" data-text="Summary">
        Summary
      </h2>

      <ul>
        <li>
          <p>
            <strong>Using a pre-trained model for feature extraction</strong>:
            When working with a small dataset, it is a common practice to take
            advantage of features learned by a model trained on a larger dataset
            in the same domain. This is done by instantiating the pre-trained
            model and adding a fully-connected classNameifier on top. The
            pre-trained model is &quot;frozen&quot; and only the weights of the
            classNameifier get updated during training. In this case, the
            convolutional base extracted all the features associated with each
            image and you just trained a classNameifier that determines the
            image className given that set of extracted features.
          </p>
        </li>
        <li>
          <p>
            <strong>Fine-tuning a pre-trained model</strong>: To further improve
            performance, one might want to repurpose the top-level layers of the
            pre-trained models to the new dataset via fine-tuning. In this case,
            you tuned your weights such that your model learned high-level
            features specific to the dataset. This technique is usually
            recommended when the training dataset is large and very similar to
            the original dataset that the pre-trained model was trained on.
          </p>
        </li>
      </ul>

      <p>
        To learn more, visit the
        <a href="https://www.tensorflow.org/guide/keras/transfer_learning">
          Transfer learning guide
        </a>
        .
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          # MIT License # # Copyright (c) 2017 François Chollet #
          IGNORE_COPYRIGHT: cleared by OSS licensing # # Permission is hereby
          granted, free of charge, to any person obtaining a # copy of this
          software and associated documentation files (the &#34;Software&#34;),
          # to deal in the Software without restriction, including without
          limitation # the rights to use, copy, modify, merge, publish,
          distribute, sublicense, # and/or sell copies of the Software, and to
          permit persons to whom the # Software is furnished to do so, subject
          to the following conditions: # # The above copyright notice and this
          permission notice shall be included in # all copies or substantial
          portions of the Software. # # THE SOFTWARE IS PROVIDED &#34;AS
          IS&#34;, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR # IMPLIED, INCLUDING
          BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, # FITNESS FOR A
          PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL # THE
          AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
          # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
          ARISING # FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
          OR OTHER # DEALINGS IN THE SOFTWARE.
        </code>
      </pre>
    </div>
  );
}

export default App;
