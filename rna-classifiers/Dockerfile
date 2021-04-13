FROM ubuntu:16.04

# Install base packages: python3
RUN apt-get update && apt-get install -y \
  git \
  sudo \
  curl \
  build-essential \
  software-properties-common \
  cmake \
  python3-pip \
  p7zip-full \
  nano

ENV MY_ROOT_APPLICATION /irene-phd
ENV MY_ROOT /root
ENV MY_CLASSPATH /root
ENV MY_DATASET /root

WORKDIR $MY_ROOT_APPLICATION

# Upgrade pip
RUN pip3 install --upgrade pip

# Basic python tools
RUN pip3 install awscli scipy numpy cython nose pandas

# Bio python tools
RUN pip3 install biopython scikit-bio

# Scikit learn
RUN pip3 install -U scikit-learn

# XGBoost
RUN pip3 install xgboost

# Fast text: https://github.com/facebookresearch/fastText
RUN git clone https://github.com/facebookresearch/fastText.git
RUN cd fastText; pip install .
RUN cd fastText; mkdir build && cd build && cmake ..; make && make install
ENV FASTTEXT_BIN_PATH $MY_ROOT_APPLICATION/fastText/build/fasttext

# Tree of Parzen Estimators Optimization: https://github.com/hyperopt/hyperopt
RUN pip3 install networkx==1.11
RUN pip3 install hyperopt

# StartSpace: https://github.com/facebookresearch/StarSpace
RUN apt-get install wget
RUN apt-get install unzip
RUN wget https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.zip
RUN unzip boost_1_63_0.zip
RUN mv boost_1_63_0 /usr/local/bin

RUN git clone https://github.com/facebookresearch/Starspace.git
RUN cd Starspace; make

# Copy Files
COPY rna-localization/ $MY_ROOT_APPLICATION/rna-localization/

# Defaults command
CMD ["bash"]
