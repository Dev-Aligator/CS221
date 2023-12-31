import { bar_plot, mo2ml } from "../assets";

const VietnameseArticle = () => {
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
            href="https://colab.research.google.com/drive/1p63AVpPfqWaOVv82l0dIPN2iHv8LeAB8"
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
          <a href="https://drive.google.com/uc?export=download&id=1p63AVpPfqWaOVv82l0dIPN2iHv8LeAB8">
            <img src="https://www.tensorflow.org/images/download_logo_32px.png" />
            Download notebook
          </a>
        </td>
      </table>

      <p>
        Trong chương này, chúng ta sẽ tìm hiểu cách thực hiện aspect category
        sentiment analysis trên{" "}
        <a
          href="https://drive.google.com/file/d/1yjZ0sDD2kAKOZK78MFqWJyaDcCSxnIqN/view?usp=drive_link"
          target="_blank"
        >
          Vietnamese restaurant reviews datasets
        </a>{" "}
        sử dụng Logistic Regression với Gradient Descent.
      </p>

      <p>
        Aspect category sentiment analysis là quy trình phân loại dữ liệu dạng
        text , dữ liệu này có thể đến từ reviews của khách hàng hoặc các bình
        luận trên mạng xã hội, thành các aspect categories cụ thể (e.g.,
        "FOOD#QUALITY," "SERVICE#GENERAL") và gán các sentiment tương ứng cho
        các aspects đó (e.g., positive, negative, neutral).
      </p>

      <p>
        Thay vì phải lập trình từng model riêng rẽ cho mỗi entity, chúng ta sẽ
        chỉ thiết kế
        <strong> một model duy nhất</strong> có khả năng phân loại sentiments
        cho các aspect categories một cách hiệu quả.
      </p>

      <p>
        Mục tiêu của chúng ta là xây dựng một sentiment analysis model đủ mạnh
        để xử lý dữ liệu text Tiếng Việt. Ta sẽ tuân theo các bước sau:
      </p>

      <ol>
        <li>Khảo sát dữ liệu</li>
        <li>
          Feature Extraction: Ta sẽ sử dụng phương pháp TF (Term Frequency)
          vectorization để số hóa dữ liệu dạng text cho phù hợp với một mô hình
          máy học
        </li>
        <li>
          Model Training
          <ul>
            <li>
              Ta sẽ thiết kế một logistic regression classifier để dự đoán
              sentiments cho các aspects khác nhau một cách đồng thời
            </li>
            <li>
              Mô hình này sẽ có khả năng phân loại text thành các aspect cụ thể
              và gán cho chúng các sentiments tương ứng (negative, neutral, hoặc
              positive)
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
import matplotlib.pyplot as plt
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
        Trong bài này, ta sẽ dùng một dataset chứa hàng nghìn reviews các nhà
        hàng bằng Tiếng Việt.
        <a href="https://drive.google.com/uc?export=download&id=1yjZ0sDD2kAKOZK78MFqWJyaDcCSxnIqN">
          {" "}
          Download
        </a>{" "}
        và giải nén file zip chứa các files csv, hoặc bạn có thể dùng files txt
        được cung cấp bởi VLSP2018 và tham khảo bước
        <a
          href="https://github.com/Dev-Aligator/CS221/blob/master/Solution/csvConverter.py"
          target="_blank"
        >
          {" "}
          này
        </a>{" "}
        để convert chúng thành tệp csv.
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
        Để load và xử lí bộ dữ liệu, ta sẽ tạo một hàm để đọc các files CSV sử
        dụng thư viện
        <a href="https://pandas.pydata.org/docs/" target="_blank">
          {" "}
          Pandas
        </a>
        . Mục tiêu là trích xuất các review text và các labels tưởng ứng để sử
        dụng cho sentiment analysis. Ta sẽ làm như sau:
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
        Sau khi đã tạo hàm để đọc và xử lí data, bước tiếp theo hãy cùng ứng
        dụng nó trên bộ training, validation, và test:
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

      <h3
        id="data_define_constants"
        data-text="Định nghĩa các Aspects và Sentiments"
      >
        Định nghĩa các Aspects và Sentiments
      </h3>
      <p>
        Dựa trên thông tin hướng dẫn từ VLSP, ta sẽ có tổng cộng 12 aspects, mỗi
        aspect ứng với 3 sentiments.
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
        Tiếp theo, chúng ta cần một hàm để convert dữ liệu dạng multi-output
        sang binary multi-label.
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
        Convert dữ liệu của chúng ta sang dạng binary multi-label sử dụng hàm
        `mo2ml` mà ta đã tạo trước đó
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          ytrain_ml = mo2ml(ytrain) <br></br>
          ydev_ml = mo2ml(ydev) <br></br>
          ytest_ml = mo2ml(ytest) <br></br>
        </code>
      </pre>

      <p>
        <img src={mo2ml} alt="png" />
      </p>

      <p>
        Ta cũng cần đảm bảo dữ liệu phải luôn ở dạng DataFrame, tạo hàm để
        convert nếu cần thiết
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `def mo2df(y):
  if isinstance(y, pd.DataFrame):
      return y
  return pd.DataFrame(y, columns=aspects)`,
          }}
        ></code>
      </pre>
      <p>
        Giờ hãy tạo một biểu đồ cột thể hiện giá trị trung bình của từng aspects
        trên bộ dữ liệu huấn luyện.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `ytrain_transposed = ytrain.T
fig, ax = plt.subplots(figsize=(12, 6))

# Iterate through each category and plot the values
for category_index, category_name in dict(enumerate(aspects)).items():
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
        Ở bước này, ta sẽ convert dữ liệu dạng text sang thuộc tính term
        frequency dùng CountVectorizer từ thư viện scikit-learn.
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
          Ta sẽ chọn cách kết hợp các tiếng (unigrams, bigrams, and trigrams) để
          bao quát nhiều thông tin hơn từ text.
          <strong>`min_df=2`:</strong> một từ phải xuất hiện ở ít nhất 2
          documents thì mới được coi là một đặc trưng.
          <strong>`max_df=0.9`:</strong> nếu một từ xuất hiện quá nhiều ( trên
          90% tổng các documents ) thì không xét từ đó.
        </span>
      </aside>

      <p>Transformation</p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          xtrain_baseCV = vectorizer.fit_transform(Xtrain) <br></br>
          xdev_baseCV = vectorizer.transform(Xdev) <br></br>
          xtest_baseCV = vectorizer.transform(Xtest) <br></br>
        </code>
      </pre>
      <p>Cùng xem kết quả sau bước transformation</p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `sample_vector = xtrain_baseCV[0]

# Retrieve terms and their indices from the vocabulary
vocab_terms = {index: term for term, index in vectorizer.vocabulary_.items()}

# For each non-zero entry in the sample vector, print the term and its frequency value
for index in sample_vector.indices:
    print(f"Term: {vocab_terms[index]}, CV value: {sample_vector[0, index]}")`,
          }}
        ></code>
      </pre>

      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Term: ảnh, CV value: 1 <br></br>
        Term: chụp, CV value: 3 <br></br>
        Term: từ, CV value: 1 <br></br>
        Term: hôm, CV value: 2 <br></br>
        Term: qua, CV value: 2 <br></br>
        Term: đi, CV value: 2 <br></br>
        Term: chơi, CV value: 1 <br></br>
        Term: với, CV value: 3 <br></br>
        Term: gia, CV value: 1 <br></br>
        Term: đình, CV value: 1 <br></br>
        Term: và, CV value: 1 <br></br>
        Term: nhà, CV value: 1 <br></br>
        Term: họ, CV value: 1 <br></br>
        Term: hàng, CV value: 1 <br></br>
        Term: đang, CV value: 1 <br></br>
        Term: sống, CV value: 1 <br></br>
        Term: tại, CV value: 1 <br></br>
        Term: sài, CV value: 1 <br></br>
        Term: gòn, CV value: 1 <br></br>
        Term: ăn, CV value: 4 <br></br>
        Term: trưa, CV value: 1 <br></br>
        Term: muộn, CV value: 1 <br></br>
        Term: ai, CV value: 1 <br></br>
        Term: cũng, CV value: 1 <br></br>
        Term: đói, CV value: 2 <br></br>
        Term: hết, CV value: 1 <br></br>
        Term: nên, CV value: 2 <br></br>
        Term: lúc, CV value: 1 <br></br>
        Term: có, CV value: 1 <br></br>
        Term: đồ, CV value: 2 <br></br>
        Term: là, CV value: 1 <br></br>
        Term: nhào, CV value: 1 <br></br>
        Term: vô, CV value: 1 <br></br>
        Term: liền, CV value: 1 <br></br>
        Term: bởi, CV value: 1 <br></br>
        Term: vậy, CV value: 1 <br></br>
        Term: mới, CV value: 1 <br></br>
        Term: quên, CV value: 1 <br></br>
        Term: các, CV value: 1 <br></br>
        Term: phần, CV value: 1 <br></br>
        Term: gọi, CV value: 1 <br></br>
        Term: thêm, CV value: 1 <br></br>
        Term: nước, CV value: 1 <br></br>
        Term: mắm, CV value: 1 <br></br>
        Term: chỉ, CV value: 1 <br></br>
        Term: món, CV value: 1 <br></br>
        Term: chính, CV value: 1 <br></br>
        Term: thôi, CV value: 1 <br></br>
        Term: quá, CV value: 1 <br></br>
        Term: không, CV value: 1 <br></br>
        Term: biết, CV value: 1 <br></br>
        Term: đánh, CV value: 1 <br></br>
        Term: giá, CV value: 1 <br></br>
        Term: kiểu, CV value: 1 <br></br>
        Term: gì, CV value: 1 <br></br>
        Term: luôn, CV value: 1 <br></br>
        Term: chọn, CV value: 1 <br></br>
        Term: cái, CV value: 1 <br></br>
        Term: này, CV value: 1 <br></br>
        Term: vì, CV value: 1 <br></br>
        Term: thấy, CV value: 1 <br></br>
        Term: nó, CV value: 1 <br></br>
        Term: lạ, CV value: 1 <br></br>
        Term: tui, CV value: 1 <br></br>
        Term: ảnh chụp, CV value: 1 <br></br>
        Term: hôm qua, CV value: 2 <br></br>
        Term: qua đi, CV value: 2 <br></br>
        Term: đi chơi, CV value: 1 <br></br>
        Term: với gia, CV value: 1 <br></br>
        Term: gia đình, CV value: 1 <br></br>
        Term: đình và, CV value: 1 <br></br>
        Term: tại sài, CV value: 1 <br></br>
        Term: sài gòn, CV value: 1 <br></br>
        Term: đi ăn, CV value: 1 <br></br>
        Term: ăn trưa, CV value: 1 <br></br>
        Term: ai cũng, CV value: 1 <br></br>
        Term: hết nên, CV value: 1 <br></br>
        Term: nên lúc, CV value: 1 <br></br>
        Term: có đồ, CV value: 1 <br></br>
        Term: đồ ăn, CV value: 2 <br></br>
        Term: ăn là, CV value: 1 <br></br>
        Term: vô ăn, CV value: 1 <br></br>
        Term: ăn liền, CV value: 1 <br></br>
        Term: vậy mới, CV value: 1 <br></br>
        Term: quên chụp, CV value: 1 <br></br>
        Term: gọi thêm, CV value: 1 <br></br>
        Term: thêm với, CV value: 1 <br></br>
        Term: với nước, CV value: 1 <br></br>
        Term: nước mắm, CV value: 1 <br></br>
        Term: món chính, CV value: 1 <br></br>
        Term: chính thôi, CV value: 1 <br></br>
        Term: đói quá, CV value: 1 <br></br>
        Term: quá nên, CV value: 1 <br></br>
        Term: nên không, CV value: 1 <br></br>
        Term: không biết, CV value: 1 <br></br>
        Term: đánh giá, CV value: 1 <br></br>
        Term: giá đồ, CV value: 1 <br></br>
        Term: ăn kiểu, CV value: 1 <br></br>
        Term: kiểu gì, CV value: 1 <br></br>
        Term: gì luôn, CV value: 1 <br></br>
        Term: chọn cái, CV value: 1 <br></br>
        Term: cái này, CV value: 1 <br></br>
        Term: này vì, CV value: 1 <br></br>
        Term: vì thấy, CV value: 1 <br></br>
        Term: thấy nó, CV value: 1 <br></br>
        Term: nó lạ, CV value: 1 <br></br>
        Term: lạ với, CV value: 1 <br></br>
        Term: với tui, CV value: 1 <br></br>
        Term: hôm qua đi, CV value: 2 <br></br>
        Term: qua đi chơi, CV value: 1 <br></br>
        Term: với gia đình, CV value: 1 <br></br>
        Term: gia đình và, CV value: 1 <br></br>
        Term: tại sài gòn, CV value: 1 <br></br>
        Term: qua đi ăn, CV value: 1 <br></br>
        Term: có đồ ăn, CV value: 1 <br></br>
        Term: đồ ăn là, CV value: 1 <br></br>
        Term: với nước mắm, CV value: 1 <br></br>
        Term: đói quá nên, CV value: 1 <br></br>
        Term: quá nên không, CV value: 1 <br></br>
        Term: nên không biết, CV value: 1 <br></br>
        Term: giá đồ ăn, CV value: 1 <br></br>
        Term: vì thấy nó, CV value: 1 <br></br>
        Term: thấy nó lạ, CV value: 1 <br></br>
      </pre>

      <h2 id="model_training" data-text="Model training">
        Model training
      </h2>

      <h3 id="evaluation_functions" data-text="Evaluation functions">
        Evaluation functions
      </h3>
      <p>
        Trước tiên, ta cần tạo các evaluation functions để kiểm tra model có
        hoạt động tốt hay không. Trong bài này, ta sẽ dùng
        'classification_report' và 'f1 score'
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `from sklearn.metrics import f1_score, classification_report


def quick_f1(y_true, y_pred):
    y_pred = mo2ml(mo2df(y_pred))
    return round(f1_score(y_true, y_pred, average='micro', zero_division=0), 4)

def evaluate(model, X, y, average='micro'):
    yb_true  = mo2ml(y)

    yb_pred  = mo2df(model.predict(X))
    yb_pred  = mo2ml(yb_pred)

    return classification_report(yb_true, yb_pred, zero_division=0)`,
          }}
        ></code>
      </pre>

      <aside className="note">
        <span>
          <strong>`The F1 score`: </strong>
          là harmonic mean của precision và recall. <br></br>
          <strong>`Classification Report`:</strong> tổng hợp các classification
          metrics dùng cho một mô hình máy học.
        </span>
      </aside>

      <h3 id="setup_training" data-text="Set up & Training">
        Set up & Training
      </h3>
      <p>
        Import các libraries cần thiết, bao gồm 'SGDClassifier' cho stochastic
        gradient descent classification, `optuna` cho hyperparameter
        optimization, `TPESampler` cho Tree-structured Parzen Estimator sampler,
        và `MultiOutputClassifier` (MOC) cho multi-output classification.
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          from sklearn.linear_model import SGDClassifier
          <br></br>
          import optuna <br></br>
          from optuna.samplers import TPESampler <br></br>
          from sklearn.multioutput import MultiOutputClassifier as MOC <br></br>
        </code>
      </pre>

      <p>
        Để đơn giản hóa bước chọn các hyperparameter, ta sẽ sử dụng Optuna. Giờ
        ta cần định nghĩa một hàm callback cho Optuna, hàm này sẽ kiểm soát và
        lưu model tốt nhất trong quá trình optimization.
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `def callback(study, trial):
  if study.best_trial.number == trial.number:
    study.set_user_attr(key='best_model', value=trial.user_attrs['model'])`,
          }}
        ></code>
      </pre>

      <p>
        Tiếp theo, tạo một objective function cho Optuna. Hàm này sẽ chứa các
        hyperparameters cần optimize, bao gồm `class_weight` and `alpha`.
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `def logistic_objective(trial):
    params = dict(
        class_weight=trial.suggest_categorical('class_weight', ['balanced', None]),
        alpha=trial.suggest_float('alpha', 1e-7, 1e-2, log=True),  # Add alpha for L2 regularization.
        random_state=5,
    )
    # This function continues...`,
          }}
        ></code>
      </pre>

      <aside className="note">
        <span>
          The `log` parameter in the `suggest_float` function suggests
          hyperparameters on a logarithmic scale.
        </span>
      </aside>

      <p>
        Giờ hãy cùng tạo một instance MultiOutputClassifier (MOC) và train nó
        bằng SGDClassifier với các hyperparameters từ trial. Model tốt nhất sẽ
        được lưu lại để sử dụng sau.
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code
          translate="no"
          dir="ltr"
          dangerouslySetInnerHTML={{
            __html: `    # logistic_objective continues here....

    clf = MOC(SGDClassifier(loss='log_loss', max_iter=200, **params))  
    clf.fit(xtrain_baseCV, ytrain)
    trial.set_user_attr(key="model", value=clf)

    y_pred = clf.predict(xdev_baseCV)
    return quick_f1(ydev_ml, y_pred)`,
          }}
        ></code>
      </pre>

      <aside className="note">
        <span>
          Việc lựa chọn loss='log_loss' thể hiện rằng logistic loss function
          (log loss) đang được sử dụng.
        </span>
      </aside>

      <p>
        Cuối cùng, set up Optuna study với sampler và optimization direction
        được chỉ định. Khởi chạy optimization process bằng cách gọi phương thức
        `optimize`
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          sampler = TPESampler(seed=221) <br></br>
          logistic_study = optuna.create_study(sampler=sampler,
          direction='maximize') <br></br>
          logistic_study.optimize(logistic_objective, n_trials=50,
          callbacks=[callback])
        </code>
      </pre>

      <aside className="note">
        <span>
          <strong>TPESampler</strong> là một trong những lựa chọn samplers của
          Optuna. Sử dụng chiến lược optimize Bayesian để tìm các hyperparameter
          một cách hiệu quả.
          <br></br>
          <strong>n_trials=50:</strong> Optuna sẽ thực hiện 50 trials để
          optimize các hyperparameters.
        </span>
      </aside>

      <p>
        Hãy cùng đánh giá model tốt nhất mà ta có trên bộ test, kiểm tra hiệu
        suất trên bộ train và dev, review các hyperparameters được sử dụng.
      </p>

      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          clf = logistic_study.user_attrs['best_model'] <br></br>
          <br></br>
          print(evaluate(clf, xtest_baseCV, ytest)) <br></br>
          <br></br>
          print('train:', quick_f1(ytrain_ml, clf4.predict(xtrain_baseCV))){" "}
          <br></br>
          print('dev: ', quick_f1(ydev_ml , clf4.predict(xdev_baseCV))){" "}
          <br></br>
          print('test:', quick_f1(ytest_ml , clf4.predict(xtest_baseCV))){" "}
          <br></br>
          <br></br>
          print(clf.estimators_[0].get_params()) <br></br>
          print(logistic_study.best_params) <br></br>
        </code>
      </pre>

      <pre
        className="tfo-notebook-code-cell-output"
        translate="no"
        dir="ltr"
        dangerouslySetInnerHTML={{
          __html: `             precision    recall  f1-score   support

          0       0.25      0.04      0.06        28
          1       0.52      0.46      0.49       175
          2       0.50      0.59      0.54       128
          3       0.00      0.00      0.00        11
          4       0.48      0.26      0.33        43
          5       0.87      0.96      0.91       403
          6       0.00      0.00      0.00        16
          7       0.00      0.00      0.00        53
          8       0.74      0.92      0.82       334
          9       0.00      0.00      0.00         3
         10       0.00      0.00      0.00        45
         11       0.00      0.00      0.00        28
         12       0.00      0.00      0.00         6
         13       0.00      0.00      0.00        11
         14       0.82      0.17      0.28        54
         15       0.00      0.00      0.00         1
         16       0.00      0.00      0.00         4
         17       0.50      0.12      0.20        41
         18       0.00      0.00      0.00         5
         19       1.00      0.04      0.08        24
         20       0.67      0.05      0.09        44
         21       0.00      0.00      0.00        13
         22       0.00      0.00      0.00         5
         23       0.58      0.69      0.63       205
         24       0.00      0.00      0.00         9
         25       0.00      0.00      0.00        62
         26       0.00      0.00      0.00        59
         27       0.67      0.08      0.14        25
         28       1.00      0.05      0.09        22
         29       0.65      0.55      0.59       128
         30       0.00      0.00      0.00        26
         31       0.50      0.02      0.04        48
         32       0.88      0.70      0.78       181
         33       1.00      0.12      0.22        16
         34       0.81      0.21      0.34        99
         35       0.36      0.08      0.13        64

  micro avg       0.71      0.52      0.60      2419
  macro avg       0.36      0.17      0.19      2419
weighted avg       0.60      0.52      0.51      2419
samples avg       0.71      0.53      0.59      2419

train: 0.9958
dev:   0.6613
test: 0.5821
{'alpha': 0.0004717578720216505, 'average': False, 'class_weight': None, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log_loss', 'max_iter': 200, 'n_iter_no_change': 5, 'n_jobs': None, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 5, 'shuffle': True, 'tol': 0.001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
{'class_weight': None, 'alpha': 0.0004717578720216505}
  `,
        }}
      ></pre>
      <p>
        Tới đây, ta đã thành công tạo một mô hình Logistic Regression để phân
        loại sentiment cho các aspect categories. Tuy nhiên, thông qua
        evaluation score - Chỉ <strong>0.5821 </strong> trên bộ test, ta có thể
        làm tốt hơn thế không ?
      </p>

      <h2 id="Optimization" data-text="Optimization">
        Optimization techniques
      </h2>

      <h3 id="data_processing" data-text="Data processing">
        Data processing
      </h3>
      <p>
        Quan sát dữ liệu các review, ta bắt gặp nhiều emojis, chữ hoa thường lẫn
        lộn, kí tự đặc biệt. Để chuẩn hóa dữ liệu, ta sẽ tạo một lớp TextCleaner
        như sau:
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
        Tuy nhiên, bạn cũng có thể bỏ qua bước này vì ảnh hưởng của nó lên hiệu
        suất cuối cùng của model là không đáng kể, hoặc bạn có thể thử một
        phương pháp khác hiệu quả hơn.
      </p>

      <h3 id="tf_idf" data-text="Sử dụng TF-IDF thay vì TF">
        Sử dụng TF-IDF thay vì TF
      </h3>
      <p>
        Chuyển sang TF-IDF sẽ cải thiện model của ta bằng cách gán trọng số lớn
        cho các từ có liên quan với nhau và giảm độ phức tạp của các đặc trưng,
        từ đó có thể phân loại các sentiment một cách chính xác và linh hoạt
        hơn.
      </p>
      <pre className="prettyprint lang-python" translate="no" dir="ltr">
        <code translate="no" dir="ltr">
          from sklearn.feature_extraction.text import TfidfVectorizer <br></br>
          <br></br>
          vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.9)
        </code>
      </pre>
      <pre className="tfo-notebook-code-cell-output" translate="no" dir="ltr">
        Term: thấy nó lạ, TF-IDF value: 0.11312900190971689 <br></br>
        Term: vì thấy nó, TF-IDF value: 0.11312900190971689 <br></br>
        Term: giá đồ ăn, TF-IDF value: 0.10900675315011185 <br></br>
        Term: nên không biết, TF-IDF value: 0.09587705653497454 <br></br>
        Term: quá nên không, TF-IDF value: 0.10580928860260153 <br></br>
        Term: đói quá nên, TF-IDF value: 0.11312900190971689 <br></br>
        Term: với nước mắm, TF-IDF value: 0.09587705653497454 <br></br>
        Term: đồ ăn là, TF-IDF value: 0.10900675315011185 <br></br>
        Term: có đồ ăn, TF-IDF value: 0.10098791670329482 <br></br>
        Term: qua đi ăn, TF-IDF value: 0.10900675315011185 <br></br>
        Term: tại sài gòn, TF-IDF value: 0.1031967698420899 <br></br>
        Term: gia đình và, TF-IDF value: 0.10900675315011185 <br></br>
        Term: với gia đình, TF-IDF value: 0.1031967698420899 <br></br>
        Term: qua đi chơi, TF-IDF value: 0.11312900190971689 <br></br>
        Term: hôm qua đi, TF-IDF value: 0.21161857720520305 <br></br>
        Term: với tui, TF-IDF value: 0.10900675315011185 <br></br>
        Term: lạ với, TF-IDF value: 0.10900675315011185 <br></br>
        Term: nó lạ, TF-IDF value: 0.10900675315011185 <br></br>
        Term: thấy nó, TF-IDF value: 0.08914228901485786 <br></br>
        Term: vì thấy, TF-IDF value: 0.10098791670329482 <br></br>
        Term: này vì, TF-IDF value: 0.09451133962697372 <br></br>
        Term: cái này, TF-IDF value: 0.08524570132764589 <br></br>
        Term: chọn cái, TF-IDF value: 0.11312900190971689 <br></br>
        Term: gì luôn, TF-IDF value: 0.1031967698420899 <br></br>
        Term: kiểu gì, TF-IDF value: 0.1031967698420899 <br></br>
        Term: ăn kiểu, TF-IDF value: 0.1031967698420899 <br></br>
        Term: giá đồ, TF-IDF value: 0.10900675315011185 <br></br>
        Term: đánh giá, TF-IDF value: 0.08667981541538645 <br></br>
        Term: không biết, TF-IDF value: 0.0783413557106385 <br></br>
        Term: nên không, TF-IDF value: 0.07712971751245862 <br></br>
        Term: quá nên, TF-IDF value: 0.07876912425132479 <br></br>
        Term: đói quá, TF-IDF value: 0.10900675315011185 <br></br>
        Term: chính thôi, TF-IDF value: 0.11312900190971689 <br></br>
        Term: món chính, TF-IDF value: 0.10098791670329482 <br></br>
        Term: nước mắm, TF-IDF value: 0.06218377264488158 <br></br>
        Term: với nước, TF-IDF value: 0.07497629648127954 <br></br>
        Term: thêm với, TF-IDF value: 0.10900675315011185 <br></br>
        Term: gọi thêm, TF-IDF value: 0.06883689218369779 <br></br>
        Term: quên chụp, TF-IDF value: 0.10098791670329482 <br></br>
        Term: vậy mới, TF-IDF value: 0.10580928860260153 <br></br>
        Term: ăn liền, TF-IDF value: 0.09326453777446292 <br></br>
        Term: vô ăn, TF-IDF value: 0.10098791670329482 <br></br>
        Term: ăn là, TF-IDF value: 0.07068839059401029 <br></br>
        Term: đồ ăn, TF-IDF value: 0.0965672943105496 <br></br>
        Term: có đồ, TF-IDF value: 0.0900670732269526 <br></br>
        Term: nên lúc, TF-IDF value: 0.09211759125252429 <br></br>
        Term: hết nên, TF-IDF value: 0.10580928860260153 <br></br>
        Term: ai cũng, TF-IDF value: 0.08827358777826548 <br></br>
        Term: ăn trưa, TF-IDF value: 0.08914228901485786 <br></br>
        Term: đi ăn, TF-IDF value: 0.05462049797329023 <br></br>
        Term: sài gòn, TF-IDF value: 0.0740099180978919 <br></br>
        Term: tại sài, TF-IDF value: 0.10098791670329482 <br></br>
        Term: đình và, TF-IDF value: 0.10900675315011185 <br></br>
        Term: gia đình, TF-IDF value: 0.07752232239881397 <br></br>
        Term: với gia, TF-IDF value: 0.09451133962697372 <br></br>
        Term: đi chơi, TF-IDF value: 0.10098791670329482 <br></br>
        Term: qua đi, TF-IDF value: 0.21161857720520305 <br></br>
        Term: hôm qua, TF-IDF value: 0.19477357306813592 <br></br>
        Term: ảnh chụp, TF-IDF value: 0.11312900190971689 <br></br>
        Term: tui, TF-IDF value: 0.07432485785130365 <br></br>
        Term: lạ, TF-IDF value: 0.05273732207956627 <br></br>
        Term: nó, TF-IDF value: 0.059123430502132986 <br></br>
        Term: thấy, TF-IDF value: 0.03976376470270375 <br></br>
        Term: vì, TF-IDF value: 0.046441557269232644 <br></br>
        Term: này, TF-IDF value: 0.032387076442000604 <br></br>
        Term: cái, TF-IDF value: 0.043162444740421864 <br></br>
        Term: chọn, TF-IDF value: 0.05934559281197689 <br></br>
        Term: luôn, TF-IDF value: 0.03384005713147784 <br></br>
        Term: gì, TF-IDF value: 0.05132675636515987 <br></br>
        Term: kiểu, TF-IDF value: 0.0551953846640886 <br></br>
        Term: giá, TF-IDF value: 0.03109910894856448 <br></br>
        Term: đánh, TF-IDF value: 0.08112345256804085 <br></br>
        Term: biết, TF-IDF value: 0.05470122597244219 <br></br>
        Term: không, TF-IDF value: 0.03130325417170216 <br></br>
        Term: quá, TF-IDF value: 0.03976376470270375 <br></br>
        Term: thôi, TF-IDF value: 0.04551677305713792 <br></br>
        Term: chính, TF-IDF value: 0.07198018206573666 <br></br>
        Term: món, TF-IDF value: 0.03259325527896704 <br></br>
        Term: chỉ, TF-IDF value: 0.04211605429394878 <br></br>
        Term: mắm, TF-IDF value: 0.05211571793048267 <br></br>
        Term: nước, TF-IDF value: 0.030760123244311387 <br></br>
        Term: thêm, TF-IDF value: 0.041046941651318676 <br></br>
        Term: gọi, TF-IDF value: 0.044252212890359015 <br></br>
        Term: phần, TF-IDF value: 0.041203889353430494 <br></br>
        Term: các, TF-IDF value: 0.045013943150734356 <br></br>
        Term: quên, TF-IDF value: 0.0722531271172703 <br></br>
        Term: mới, TF-IDF value: 0.04577496350093349 <br></br>
        Term: vậy, TF-IDF value: 0.05353560950395495 <br></br>
        Term: bởi, TF-IDF value: 0.08594482446734754 <br></br>
        Term: liền, TF-IDF value: 0.07565876746111884 <br></br>
        Term: vô, TF-IDF value: 0.061258988432786855 <br></br>
        Term: nhào, TF-IDF value: 0.11312900190971689 <br></br>
        Term: là, TF-IDF value: 0.024703556181286373 <br></br>
        Term: đồ, TF-IDF value: 0.08668904131683706 <br></br>
        Term: có, TF-IDF value: 0.025451157146521262 <br></br>
        Term: lúc, TF-IDF value: 0.05171580782757912 <br></br>
        Term: nên, TF-IDF value: 0.06553473311560175 <br></br>
        Term: hết, TF-IDF value: 0.041717995865396706 <br></br>
        Term: đói, TF-IDF value: 0.15504464479762795 <br></br>
        Term: cũng, TF-IDF value: 0.028406256377782924 <br></br>
        Term: ai, TF-IDF value: 0.056603061139257685 <br></br>
        Term: muộn, TF-IDF value: 0.09451133962697372 <br></br>
        Term: trưa, TF-IDF value: 0.06521165909550117 <br></br>
        Term: ăn, TF-IDF value: 0.07683878126855012 <br></br>
        Term: gòn, TF-IDF value: 0.0740099180978919 <br></br>
        Term: sài, TF-IDF value: 0.07340007363920893 <br></br>
        Term: tại, TF-IDF value: 0.06088022148275457 <br></br>
        Term: sống, TF-IDF value: 0.07464687549171974 <br></br>
        Term: đang, TF-IDF value: 0.0677905017372247 <br></br>
        Term: hàng, TF-IDF value: 0.05553455273377999 <br></br>
        Term: họ, TF-IDF value: 0.08594482446734754 <br></br>
        Term: nhà, TF-IDF value: 0.053167229783046555 <br></br>
        Term: và, TF-IDF value: 0.02993178719061098 <br></br>
        Term: đình, TF-IDF value: 0.07432485785130365 <br></br>
        Term: gia, TF-IDF value: 0.057657858263559986 <br></br>
        Term: với, TF-IDF value: 0.10543082488081452 <br></br>
        Term: chơi, TF-IDF value: 0.08062062266163728 <br></br>
        Term: đi, TF-IDF value: 0.07913067133800834 <br></br>
        Term: qua, TF-IDF value: 0.11914250776873991 <br></br>
        Term: hôm, TF-IDF value: 0.1269356831431639 <br></br>
        Term: từ, TF-IDF value: 0.04777546687904782 <br></br>
        Term: chụp, TF-IDF value: 0.23377796406159154 <br></br>
        Term: ảnh, TF-IDF value: 0.07921005694723088 <br></br>
      </pre>

      <p>
        Sau khi áp dụng 2 phương pháp đã đề cập ở trên, ta có được kết quả cuối
        cùng như sau:
      </p>

      <pre
        className="tfo-notebook-code-cell-output"
        translate="no"
        dir="ltr"
        dangerouslySetInnerHTML={{
          __html: `train: 0.9969
dev:   0.6629
test: 0.6224
{'alpha': 6.397620627963636e-05, 'average': False, 'class_weight': 'balanced', 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log_loss', 'max_iter': 200, 'n_iter_no_change': 5, 'n_jobs': None, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 5, 'shuffle': True, 'tol': 0.001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
{'class_weight': 'balanced', 'alpha': 6.397620627963636e-05}`,
        }}
      ></pre>
      <p>
        Bài tutorial kết thúc ở đây, hãy thử download notebook này và thử nghiệm
        với các phương pháp optimization của bạn.
      </p>
    </div>
  );
};

export default VietnameseArticle;
