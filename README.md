## Iris Dataset Machine Learning with Tensorflow
This is a quick project that is often seen as one of the few "Hello World" projects of machine learning. This uses the Iris flower dataset that was created by Ronald Fisher to train and test a neural network to be able to classify the species of Iris flower based on the features passed in (sepal width, sepal length, petal width, and petal length). The overall objective is to reach greater than 90% test accuracy which this does after only a few training procedures.
#### Python Version 3.6.7

#### Tensorflow build note
It is recommended that you use the Tensorflow package that corresponds to the hardware you intend to designate the workload to
The base `pip install tensorflow` installs a version that is designed to work with as many builds as possible
by visiting https://github.com/lakshayg/tensorflow-build you can find the package that works best with your GPU/CPU as well as the corresponding Tensorflow release. In executing my script, I initially had AVX2 disabled since the base build does not utilize it in the event that a CPU is not capable of it. By using `pip install --ignore-installed <URL of package>` it will overwrite your pre-existing Tensorflow build and it's dependencies with your optimized specifications.
If you are unsure about your specifications, the base build will work fine, however in my case, a quick google search of my CPU brought me to: https://ark.intel.com/products/88193/Intel-Core-i5-6200U-Processor-3M-Cache-up-to-2-80-GHz- which under the "Instruction Set Extensions" says that SSE4.1, SSE4.2, and AVX2 are all compatible with this CPU.
I then simply downloaded the new Tensorflow build that corresponded to my CPU which was an AVX2 enabled 1.11.0 win64 build. Using pip within a virtual environment will then do the rest of the work for you.

### Setup Instructions
This script was created using virtualenv for package management
1. Path to your desired project location `cd /to/your/project`
2. My base python extension in 3.7.1, so in order to create a virtual environment compatible with Tensorflow, I needed to download Python 3.6.x and create the virtual environment with that specified path to Python 3.6.x. From my project location I used the command: `virtualenv -p /path/to/python36/python venv`
3. Finally, path into the /venv/Scripts/ directory and activate it by typing the `activate` command.
4. If you plan to use the base build of Tensorflow (see "Tensorflow Build Notes" for information), then you can simply run `pip install -r requirements.txt` otherwise it is recommended you run `pip install -r non-base-TF.txt` followed by `pip install --ignore-installed <custom_tensorflow.whl_path>` in order to install Tensorflow with a custom build. This will also overwrite any previously installed packages that have differnt dependencies depending on the Tensorflow build you selected.
5. At this point, all that is needed to do it path to the /venv/app/ folder and run app.py with: `python app.py`. This will download the iris dataset as a .csv file into your /app/ folder and will then use it to train and test the neural network using Tensorflow

### Pip list
__Package__ == __Version__

* absl-py == 0.6.1
* astor == 0.7.1
* certifi == 2018.10.15
* chardet == 3.0.4
* cycler == 0.10.0
* gast == 0.2.0
* grpcio == 1.16.0
* h5py == 2.8.0
* idna == 2.7
* Keras-Applications == 1.0.6
* Keras-Preprocessing == 1.0.5
* kiwisolver == 1.0.1
* Markdown == 3.0.1
* matplotlib == 3.0.1
* numpy == 1.15.4
* pandas == 0.23.4
* pip == 18.1
* protobuf == 3.6.1
* pyparsing == 2.3.0
* python-dateutil == 2.7.5
* pytz == 2018.7
* requests == 2.20.0
* scikit-learn == 0.20.0
* scipy == 1.1.0
* seaborn == 0.9.0
* setuptools == 40.5.0
* six == 1.11.0
* sklearn == 0.0
* tensorboard == 1.12.0
* tensorflow == 1.12.0
* termcolor == 1.1.0
* urllib3 == 1.24.1
* Werkzeug == 0.14.1
* wheel == 0.32.2
