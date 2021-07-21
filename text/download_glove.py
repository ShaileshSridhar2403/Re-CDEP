def download_glove():
    import os
    os.system('wget http://nlp.stanford.edu/data/glove.6B.zip')
    os.system('unzip -q glove.6B.zip')
    os.system('mv glove.6B glove')


download_glove()
