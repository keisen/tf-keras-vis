FROM ubuntu

# Default Settings
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHON_VERSION 3.6.10
ENV TF_VERSION 2.1.0
ENV TF_KERAS_VIS_VERSION 0.1.0

# Install essential libraries
RUN apt update                                           \
    && apt upgrade -y                                    \
    && apt install -y --no-install-recommends            \
          bash-completion curl ca-certificates           \
    && apt install -y --no-install-recommends            \
          git build-essential                            \
          libffi-dev libssl-dev zlib1g-dev               \
          libbz2-dev libreadline-dev libsqlite3-dev      \
    && apt autoremove -y                                 \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl -L https://pyenv.run | /bin/bash
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH

RUN pyenv install -v $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION
ENV PATH $PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$PYENV_ROOT/versions/$PYTHON_VERSION/lib

# Setting for jupyter
RUN mkdir /root/.jupyter
RUN touch /root/.jupyter/jupyter_notebook_config.py
RUN echo 'c.NotebookApp.allow_root = True' >> /root/.jupyter/jupyter_notebook_config.py
RUN echo 'c.NotebookApp.ip = "0.0.0.0"' >> /root/.jupyter/jupyter_notebook_config.py
RUN echo 'c.NotebookApp.open_browser = False' >> /root/.jupyter/jupyter_notebook_config.py
RUN echo 'c.NotebookApp.token = ""' >> /root/.jupyter/jupyter_notebook_config.py

# Install essential python libraries
RUN pip install --no-cache-dir --upgrade pip setuptools
RUN pip install --no-cache-dir \
      tf-keras-vis==$TF_KERAS_VIS_VERSION \
      tensorflow==$TF_VERSION \
      numpy scipy imageio pillow \
      jupyterlab matplotlib
