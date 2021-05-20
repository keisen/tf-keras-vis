FROM tensorflow/tensorflow:2.4.1

# Default ENV Settings
ARG TF_KERAS_VIS_VERSION=0.6.0
ARG JUPYTER_ALLOW_IP="0.0.0.0"
ARG JUPYTER_TOKEN=""

# Setting for jupyter
RUN export JUPYTER_HOME=/etc/jupyter                               && \
    export JUPYTER_CONF=$JUPYTER_HOME/jupyter_notebook_config.py   && \
    mkdir -p $JUPYTER_HOME                                         && \
    touch $JUPYTER_CONF                                            && \
    echo 'c.NotebookApp.allow_root = True' >> $JUPYTER_CONF        && \
    echo "c.NotebookApp.ip = '$JUPYTER_ALLOW_IP'" >> $JUPYTER_CONF && \
    echo "c.NotebookApp.token = '$JUPYTER_TOKEN'" >> $JUPYTER_CONF && \
    echo "c.NotebookApp.terminado_settings =                          \
                {'shell_command': ['/bin/bash']}" >> $JUPYTER_CONF

# Install essential python libraries
RUN pip install --no-cache-dir \
      tf-keras-vis[develop,examples]==$TF_KERAS_VIS_VERSION

CMD jupyter lab