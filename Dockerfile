FROM b.gcr.io/tensorflow/tensorflow
RUN pip install --upgrade pip
RUN pip install -U pandas
RUN easy_install scipy
RUN pip install --upgrade scipy 
RUN pip install -U scikit-learn
RUN pip install --upgrade --user ipython
RUN mkdir /notebook
VOLUME /notebook
EXPOSE 8888 6006
CMD jupyter notebook /notebook