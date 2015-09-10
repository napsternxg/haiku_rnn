# haiku_rnn
Generating Haikus Using Char-RNN

Developed using Andrej Karpathy's char-rnn code implementation in Keras. 

## Dependencies
 * Theano
 * Keras
 * Numpy
 * IPython Notebook (optional, for notebook)
 * BeautifulSoup (optional, for data download)

## Data
Data collection details can be found in `Download Haiku.ipynb`
Final Data is downloaded in `haiku_all.txt` with each line containing 1 haiku whose lines are seperated by tabs

## Run Program
* **Standalone:** You can run the program using `python haiku_gen.py` 
* **Ipython:** This is the suggested method run the following commands
```python
%run -i haiku_gen.py # Will run the script and give access to model and data
# Will train the model. Takes almost 10 hours on my Intel Xeon CPU (Using GPU should speed it up)
# Number of epochs need to be tuned 
history = model.fit(X, y, batch_size=10000, nb_epoch=20) 
generate_from_model(model) # Generate random data from the model
```


