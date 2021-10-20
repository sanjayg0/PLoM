## Dependencies and requirements

### Main dependencies
* Python - Implementation language
* numpy and scipy - Scientific computing with Python
* pandas and tables - Input and output text data files, storing and managing data
* matplotlib - Visualization
* pytest - Testing
* hdf5 - Database server

### Required versions


  Dependency   |  Version(s)  
---------------|--------------
  [Python](https://www.python.org/downloads/)       |     3.7+
  [numpy](https://numpy.org/install/)        |  >= 1.19.0
  [scipy](https://www.scipy.org/install.html)        |  >= 1.7.1
  [pandas](https://pandas.pydata.org/docs/getting_started/install.html)      |  >= 1.2.1
  [tables](https://www.pytables.org/usersguide/installation.html)       |  >= 3.6.1
  [matplotlib](https://matplotlib.org/stable/users/installing.html)   |  >= 3.4.2
  [pytest](https://docs.pytest.org/en/6.2.x/getting-started.html) |  >= 6.2.5
  [hdf5](https://www.hdfgroup.org/solutions/hdf5/) | 

To install the hdf5, users please download the [pre-built distributions](https://www.hdfgroup.org/downloads/hdf5); macOS homebrew users can also install via:
```shell
brew install hdf5
```

To install tables for python3.9 on macOS, please first install the c-blosc:
```shell
brew install c-blosc
```

Once the hdf5 (and c-blosc if needed) installed, please see [requirements.txt](../requirements.txt) for a complete list of Python dependencies. For first-time users, please run the following command 
from the **PLoM** root to install the dependencies:

```shell
pip install -r requirements.txt
```
or
```shell
pip3 install -r requirements.txt
```
