import streamlit as st

# 1. File Input:
def file_selector(self):
   file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
   if file is not None:
      data = pd.read_csv(file)
      return data
   else:
      st.text("Please upload a csv file")

def set_features(self):
   self.features = st.multiselect("Please choose the features including target variable that go into the model", self.data.columns )

# 2. Data Preparation:
def prepare_data(self, split_data, train_test):
   # Reduce data size
   data = self.data[self.features]
   data = data.sample(frac = round(split_data/100,2))

   # Impute nans with mean for numeris and most frequent for categoricals
   cat_imp = SimpleImputer(strategy="most_frequent")
   if len(data.loc[:,data.dtypes == 'object'].columns) != 0:
      data.loc[:,data.dtypes == 'object'] = cat_imp.fit_transform(data.loc[:,data.dtypes == 'object'])
      imp = SimpleImputer(missing_values = np.nan, strategy="mean")
      data.loc[:,data.dtypes != 'object'] =      imp.fit_transform(data.loc[:,data.dtypes != 'object'])

       # One hot encoding for categorical variables
      cats = data.dtypes == 'object'
      le = LabelEncoder()
      for x in data.columns[cats]:
         data.loc[:,x] = le.fit_transform(data[x])
         onehotencoder = OneHotEncoder()      
         data.loc[:,~cats].join(pd.DataFrame(data=onehotencoder.
         fit_transform (data.loc[:,cats]).toarray(), columns= 
         onehotencoder.get_feature_names()))

         # Set target column
target_options = data.columns
self.chosen_target = st.sidebar.selectbox("Please choose target column", (target_options))

# Standardize the feature data
X = data.loc[:, data.columns != self.chosen_target]
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = data.loc[:, data.columns != self.chosen_target].columns
y = data[self.chosen_target]

# Train test split
try:
   self.X_train, self.X_test, self.y_train, self.y_test =     train_test_split(X, y, test_size=(1 - train_test/100), random_state=42)
except:
   st.markdown('<span style="color:red">With this amount of data and   split size the train data will have no records, <br /> Please change reduce and split parameter <br /> </span>', unsafe_allow_html=True)

"""
3. Choose the algorithm:
Now we will add another "selectbox" on the sidebar to enable algorithm selection like "Classification" or "Regression". 
Then, depending on previous selection we will add new selectbox with available methods for the chosen algorithm type.
"""

def set_classifier_properties(self):
   self.type = st.sidebar.selectbox("Algorithm type", ("Classification", "Regression"))
   if self.type == "Regression":
      self.chosen_classifier = st.sidebar.selectbox("Please choose a classifier", (‘Random Forest’, ‘Linear Regression’, ‘Neural Network’))
   if self.chosen_classifier == ‘Random Forest’:
      self.n_trees = st.sidebar.slider(‘number of trees’, 1, 1000, 1)
   elif self.chosen_classifier == ‘Neural Network’:
      self.epochs = st.sidebar.slider(‘number of epochs’, 1 ,100 ,10)
      self.learning_rate = float(st.sidebar.text_input(‘learning rate:’, ‘0.001’))
   elif self.type == "Classification":
      self.chosen_classifier = st.sidebar.selectbox("Please choose a classifier", (‘Logistic Regression’, ‘Naive Bayes’, ‘Neural Network’))
   if self.chosen_classifier == ‘Logistic Regression’:
      self.max_iter = st.sidebar.slider(‘max iterations’, 1, 100, 10)
   elif self.chosen_classifier == ‘Neural Network’:
      self.epochs = st.sidebar.slider(‘number of epochs’, 1 ,100 ,10)
      self.learning_rate = float(st.sidebar.text_input(‘learning rate:’, ‘0.001’))
      self.number_of_classes = int(st.sidebar.text_input(‘Number of classes’, ‘2’))

# 4. Make the prediction:
def predict(self, predict_btn):
   if self.type == "Regression":
      if self.chosen_classifier == ‘Random Forest’:
         self.alg = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=self.n_trees)
         self.model = self.alg.fit(self.X_train, self.y_train)
predictions = self.alg.predict(self.X_test)
         self.predictions_train = self.alg.predict(self.X_train)
         self.predictions = predictions
      elif self.chosen_classifier==’Linear Regression’:
         self.alg = LinearRegression()
         self.model = self.alg.fit(self.X_train, self.y_train)
         predictions = self.alg.predict(self.X_test)
         self.predictions_train = self.alg.predict(self.X_train)
         self.predictions = predictions
      
      elif self.chosen_classifier==’Neural Network’:
         model = Sequential()
         model.add(Dense(500, input_dim = len(self.X_train.columns), activation=’relu’,))
         model.add(Dense(50, activation=’relu’))
         model.add(Dense(50, activation=’relu’))
         model.add(Dense(1))
         model.compile(loss= "mean_squared_error" ,    optimizer=’adam’, metrics=["mean_squared_error"])
         self.model = model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=40)
         self.predictions = model.predict(self.X_test)
         self.predictions_train = model.predict(self.X_train)
   elif self.type == "Classification":
      if self.chosen_classifier == ‘Logistic Regression’:
          self.alg = LogisticRegression()
          self.model = self.alg.fit(self.X_train, self.y_train)
          predictions = self.alg.predict(self.X_test)
          self.predictions_train = self.alg.predict(self.X_train)
          self.predictions = predictions
      elif self.chosen_classifier==’Naive Bayes’:
         self.alg = GaussianNB()
         self.model = self.alg.fit(self.X_train, self.y_train)
         predictions = self.alg.predict(self.X_test)
         self.predictions_train = self.alg.predict(self.X_train)
         self.predictions = predictions 
      elif self.chosen_classifier==’Neural Network’:
         model = Sequential()
         model.add(Dense(500, input_dim = len(self.X_train.columns), activation=’relu’))
         model.add(Dense(50, activation=’relu’))
         model.add(Dense(50, activation=’relu’))
         model.add(Dense(self.number_of_classes, activation=’softmax’))
         optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
         model.compile(optimizer=’adam’, loss=’sparse_categorical_crossentropy’, metrics=[‘accuracy’])
         self.model = model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=40)
         self.predictions = model.predict_classes(self.X_test)
         self.predictions_train = model.predict_classes(self.X_train)

def get_metrics(self):
   self.error_metrics = {}
   if self.type == 'Regression':
      self.error_metrics['MSE_test'] = mean_squared_error(self.y_test, self.predictions)
      self.error_metrics['MSE_train'] = mean_squared_error(self.y_train, self.predictions_train)
      return st.markdown('### MSE Train: ' + str(round(self.error_metrics['MSE_train'], 3)) +
' -- MSE Test: ' + str(round(self.error_metrics['MSE_test'], 3)))
   elif self.type == 'Classification':
      self.error_metrics['Accuracy_test'] = accuracy_score(self.y_test, self.predictions)
      self.error_metrics['Accuracy_train'] = accuracy_score(self.y_train, self.predictions_train)
      return st.markdown('### Accuracy Train: ' + str(round(self.error_metrics['Accuracy_train'], 3)) +
' -- Accuracy Test: ' +  str(round(self.error_metrics['Accuracy_test'], 3)))
def plot_result(self):
   output_file("slider.html")
   s1 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
   s1.circle(self.result_train.index, self.result_train.Actual_Train, size=12, color=Set3[5][3], alpha=1)
   s1.triangle(self.result_train.index, self.result_train.Prediction_Train, size=12, color=Set3[5][4], alpha=1)
   tab1 = Panel(child=s1, title="Train Data")
   if self.result.Actual is not None:
      s2 = figure(plot_width=800, plot_height=500, background_fill_color="#fafafa")
      s2.circle(self.result.index, self.result.Actual, size=12, color=Set3[5][3], alpha=1)
      s2.triangle(self.result.index, self.result.Prediction, size=12, color=Set3[5][4], alpha=1)
      tab2 = Panel(child=s2, title="Test Data")
      tabs = Tabs(tabs=[ tab1, tab2 ])
   else:
      tabs = Tabs(tabs=[ tab1])
st.bokeh_chart(tabs)